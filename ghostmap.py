import numpy
import math
from matplotlib.delaunay.triangulate import Triangulation

# PIL
from PIL import Image
from PIL import ImageDraw

class PointList2D(object):
    """Abstract class for a set of data points in 2D space."""
    def __init__(self):
        #super(PointList, self).__init__()
        # numpy arrays of the point locations
        self.x = None
        self.y = None
        self.z = None # the value of the point
        
        self.xRange = [None, None]
        self.yRange = [None, None]

class AccretionGenerator(object):
    """Baseclass for EqualSNGenerator and EqualMassGeneator"""
    def __init__(self):
        super(AccretionGenerator, self).__init__()
    
    def reassign_bad_bins(self, binNums, x, y, numPixels):
        """
        Reassign points in 'bad' bins that were produced by the prior
        bin accretion loop. Any points that could not be binned successfully
        are accreted onto nearby existing bins.
        """
        # Obtain the number of good bins from the original binning
        nBins = numpy.max(binNums)
        print "Initial number of bins: %i" % nBins
        
        # The first task is to re-order the bin numbering to get rid of holes
        # caused by bad bins.
        newBinNumber = 0 # indexes in the new-bin
        binIndices = [] # an order list whose elements are the tuples containing
                        # indexes to the bins
        
        for i in xrange(nBins):
            j = i + 1 # j is the actual old bin number
            indices = numpy.where(binNums==j)[0] # elements in bin j
            if len(indices)==0:
                # in this case, the bin is empty because it was a failure
                # (harsh words...)
                continue
            else:
                # in this case the bin is good, lets re-order it and keep
                # references to its indices
                newBinNumber = newBinNumber + 1
                binNums[indices] = newBinNumber # this re-orders the bins
                binIndices = binIndices + [indices] # hold onto the index list
                # note that the newBinNumber in binNums[indices] is the index
                # to binIndices
        
        nNodes = newBinNumber # updated number of nodes
        
        # Calculate the geometric centroid of each bin
        xNode = numpy.zeros([nNodes], dtype=numpy.float)
        yNode = numpy.zeros([nNodes], dtype=numpy.float)
        for i in xrange(nNodes):
            indices = binIndices[i]
            xNode[i] = numpy.average(x[indices])
            yNode[i] = numpy.average(y[indices])
        
        # Now reassign all unbinned pixels to the nearest good bin as
        # judged by the distance to the bin's centroid
        unbinned = numpy.where(binNums==0)[0]
        numUnbinned = len(unbinned)
        print "Reassigning %i bins" % numUnbinned
        for i in xrange(numUnbinned):
            dists = (x[unbinned[i]] - xNode)**2 + (y[unbinned[i]] - yNode)**2
            k = numpy.argmin(dists) # get the closest node
            binNums[unbinned[i]] = k+1 # assign pixel to closest node
        
        # All data elements have now been binned. Now compute the centroids for
        # these final bins. These centroids can be used as generators in a
        # tessellation procedure.
        binIndices = [] # reset the bin indices from above
        print "Created %i bins" % numpy.max(binNums)
        for i in xrange(nNodes):
            indices = numpy.where(binNums==(i+1))[0]
            # TODO check if the first statement in the if is ever used
            if len(indices) == 0:
                print "No indices for bin %i" % i
                continue
            else:
                binIndices = binIndices + [indices]
        
        # Calculation of centroids
        for i in xrange(nNodes):
            indices = binIndices[i]
            xNode[i] = numpy.average(x[indices])
            yNode[i] = numpy.average(y[indices])
        
        return (indices, binNums, xNode, yNode)
    
    def get_nodes(self):
        """Returns the x and y positions of the Voronoi nodes."""
        return self.xNode, self.yNode
    
    def get_node_membership(self):
        """Returns an array, the length of the input data arrays in
        `tessellate()`, which have indices into the node arrays of
        `get_nodes()`.
        """
        return self.binNums

class EqualSNGenerator(AccretionGenerator):
    """Generates Voronoi nodes from a data so that each Voronoi cell has a
    minimum collective S/N.
    """
    def __init__(self):
        super(EqualSNGenerator, self).__init__()
        self.xNode = None
        self.yNode = None
        self.binNums = None
    
    def generate_nodes(self, xPoints, yPoints, signal, noise, targetSN):
        """Accretes points to creates nodes hosting the target level of S/N."""
        nPoints = len(xPoints)
        self.targetMeasure = targetSN
        
        # binNums contains the bin identification number of each point
        binNums = numpy.zeros([nPoints], dtype=numpy.uint32)
        
        # good contains 1 if the pixel is in a good bin; otherwise its 0
        good = numpy.zeros([nPoints], dtype=numpy.uint32)
        
        # start bin accretion from the pixel with the highest SN
        # currentBin is a vector initialized with the first pixel
        currentBin = numpy.array([signal.argmax()], dtype=numpy.uint32)
        currentSN = signal.max() # initialize S/N of the current bin
        
        print "Accreting points into bins (SN regime)"
        # Pixel accretion for loop
        for ind in xrange(1,nPoints+1): # start from one and go to numPixels
            #print "Accreting bin #%i" % ind
            
            # initialize a new bin
            binNums[currentBin] = ind # here current bin is still made of one pixel
            xBar = xPoints[currentBin]
            yBar = yPoints[currentBin]
            
            while 1:
                # unbinned holds the indexes of all pixels yet to be binned
                unbinned = numpy.where(binNums==0)[0]
                m = unbinned.size
                
                # test if there are no pixels left to bin
                if m == 0:
                    break
                
                # find the unbinned pixel closest to the centroid of the current bin
                dist = (xPoints[unbinned]-xBar)**2 + (yPoints[unbinned]-yBar)**2
                k = numpy.argmin(dist)
                
                # Add the new unbinned[k] pixel to the "next bin"
                nextBin = numpy.concatenate((currentBin, numpy.array([unbinned[k]], dtype=numpy.uint32)))
                
                # Hang onto the "old" measure before computing new one
                oldSN = currentSN
                currentSN = numpy.sum(signal[nextBin]) / numpy.sqrt(numpy.sum(noise[nextBin]**2))
                
                # Now test whether the CANDIDATE pixel is connected to the
                # currentBin, whether the POSSIBLE new bin is round enough
                # and whether the resulting S/N would get closer to targetSN
                if abs(currentSN-self.targetMeasure) > abs(oldSN-self.targetMeasure):
                    # if candidate bin passed the above tests, then it "failed"
                    # and is not a good bin
                    if oldSN > 0.8*self.targetMeasure:
                        # but if the failed candidate bin, before adding the next
                        # pixel, had a good enough SN ratio (ie, 0.8 of targetSN), 
                        # then we accept it anyways, and break
                        good[currentBin] = 1
                    # regardless, we break the while loop
                    break
                
                # In the event that above tests were negative with the candidate
                # pixel, then we accept the candidate pixel, add it to the current
                # bin, and continue accreting more pixels
                binNums[unbinned[k]] = ind
                currentBin = nextBin
                
                # Update the centroid of the current bin
                xBar = numpy.average(xPoints[currentBin])
                yBar = numpy.average(yPoints[currentBin])
            
            # Recompute lists of what has been binned, and not binned
            unbinned = numpy.where(binNums == 0)[0]
            binned = numpy.where(binNums != 0)[0]
            if unbinned.size == 0:
                # if there are no more unbinned pixels, just break...
                break
            
            # Compute geometric centroid of all binned pixels
            xBar = numpy.average(xPoints[binned])
            yBar = numpy.average(yPoints[binned])

            # Now find the closest unbinned pixel to the centroid of all the
            # binned pixels and start a new bin from there
            dist = (xPoints[unbinned] - xBar)**2 + (yPoints[unbinned] - yBar)**2
            k = numpy.argmin(dist) # k is index to unbinned pixel closest to the
                                   # centroid of all binned pixels
            # the next bin is initially made of one pixel
            currentBin = numpy.array([unbinned[k]], dtype=numpy.uint32)
            currentSN = signal[currentBin] / noise[currentBin] # S/N of new bin
            # loop now repeats with new bin
        
        # If a pixel was assigned a bin number in binNums, but that bin was not
        # successful as determined by the "good" array, then reset the binNum to
        # zero, indicating it is still unbinned.
        binNums = binNums * good
        
        print "Finished accreting bins for target S/N"
        
        # Clean up the binning, save results
        indices, binNums, xBin, yBin = self.reassign_bad_bins(binNums,
                xPoints, yPoints, nPoints)
        self.xNode = xBin
        self.yNode = yBin
        self.binNums = binNums

class EqualMassGenerator(AccretionGenerator):
    """Generates Voronoi nodes from data points so that each Voronoi cell has
    a minimum collective mass.
    
    The generateNodes currently assumes that each point has a unitary mass. This
    should be fixed in the future...
    """
    def __init__(self):
        super(EqualMassGenerator, self).__init__()
        self.xNode = None
        self.yNode = None
        self.binNums = None
    
    def generate_nodes(self, xPoints, yPoints, mass, targetMass):
        """Generates tessellation nodes from the points so that each node
        holds the minimum accumulated mass.
        
        .. note:: the algorithm currently ignores mass;
           each point is given a mass of 1.
        """
        nPoints = len(xPoints)
        self.targetMeasure = targetMass
        
        # binNums contains the bin identification number of each point
        binNums = numpy.zeros([nPoints], dtype=numpy.uint32)
        # good contains 1 if the pixel is in a good bin; otherwise its 0
        good = numpy.zeros([nPoints], dtype=numpy.uint32)
        
        # Start bin accretion of the point closest to the center of the data
        # distribution (ie, near centre of image)
        xCentre = (xPoints.max() - xPoints.min()) / 2.
        yCentre = (yPoints.max() - yPoints.min()) / 2.
        centreDist = (xPoints-xCentre)**2 + (yPoints-yCentre)**2
        currentBin = numpy.array([centreDist.argmin()], dtype=numpy.uint32)
        currentNumPoints = 1 # initial 1 pixel/particle in the currentBin
        
        print "Accreting bins (density regime)"
        # Pixel accretion loop
        for ind in xrange(1, nPoints+1, 1):
            # initialize a new bin
            binNums[currentBin] = ind # here current bin is still made of one pixel
            xBar = xPoints[currentBin] # initialize the centroid of the bin
            yBar = yPoints[currentBin]
            
            while True:
                # unbinned holds the indices of all pixels yet to be binned
                unbinned = numpy.where(binNums==0)[0]
                m = unbinned.size
                
                # test if there are no points left to bin
                if m == 0:
                    break
                
                # find the unbinned point closest to the centroid of the current bin
                dist = (xPoints[unbinned]-xBar)**2 + (yPoints[unbinned]-yBar)**2
                k = numpy.argmin(dist)
                
                # Add the new unbinned[k] pixel to the "next bin"
                nextBin = numpy.concatenate((currentBin, numpy.array([unbinned[k]], dtype=numpy.uint32)))
                currentNumPoints = currentNumPoints + 1
                
                # Have we accreted enough stars onto the bin?
                if currentNumPoints >= self.targetMeasure:
                    # if yes, then the bin is fully formed!
                    good[currentBin] = 1
                    break
                else:
                    # if the bin not yet fully formed, we accept the candidate
                    # pixel, add it to the currentBin, and continue accretion
                    binNums[unbinned[k]] = ind
                    currentBin = nextBin
                    
                    # update the centroid of the current bin
                    xBar = numpy.average(xPoints[currentBin])
                    yBar = numpy.average(yPoints[currentBin])
            
            # Finally, recompute lists of what has been binned, and not binned
            unbinned = numpy.where(binNums == 0)[0]
            binned = numpy.where(binNums != 0)[0]
            if unbinned.size == 0:
                # if there are no more unbinned pixels, just break
                break
            
            # Compute geometric centroid of all binned pixels
            xBar = numpy.average(xPoints[binned])
            yBar = numpy.average(yPoints[binned])
            
            # Find the closest unbinned pixel to the centroid of all the
            # binned pixels and start a new bin from there
            dist = (xPoints[unbinned]-xBar)**2 + (yPoints[unbinned]-yBar)**2
            k = numpy.argmin(dist)
            # the next bin is initially made of just one pixel
            currentBin = numpy.array([unbinned[k]], dtype=numpy.uint32)
            currentNumPoints = 1
        
        # If a pixel was assigned a bin number in binNums, but that bin was not
        # successful as determined by the "good" array, then reset the binNum
        # to zero, indicating it is still unbinned.
        binNums = binNums * good
        
        print "Finished accretion"
        
        # Clean up the binning, save results
        indices, binNums, xBin, yBin = self.reassign_bad_bins(binNums,
                xPoints, yPoints, nPoints)
        self.xNode = xBin
        self.yNode = yBin
        self.binNums = binNums

class CVTessellation(object):
    """Uses Lloyd's algorithm to assign data points to Voronoi bins so that each
    bin has an equal mass.
    """
    def __init__(self):
        super(CVTessellation, self).__init__()
        self.xNode = None
        self.yNode = None
        self.vBinNum = None
    
    def tessellate(self, xPoints, yPoints, densPoints, preGenerator=None):
        """ Computes the centroidal voronoi tessellation itself.
        :param xPoints: array of cartesian `x` locations of each data point.
        :param yPoints: array of cartesian `y` locations of each data point.
        :param densPoints: array of the density of each point. For an equal-S/N
            generator, this should be set to (S/N)**2. For an equal number generator
            this can be simple an array of ones.
        :param preGenerator: an optional node generator already computed from
            the data.
        """
        nPoints = len(xPoints)
        
        # Obtain pre-generator node coordinates
        if preGenerator is not None:
            xNode, yNode = preGenerator.get_nodes()
        else:
            # Make a null set of generators, the same as the voronoi points themselves
            xNode = xPoints.copy()
            yNode = yPoints.copy()
        nNodes = len(xNode)
        
        # vBinNum holds the Voronoi bin numbers for each data point
        vBinNum = numpy.zeros(nPoints, dtype=numpy.uint32)
        
        iters = 1
        while 1:
            xNodeOld = xNode.copy()
            yNodeOld = yNode.copy()
            
            for j in xrange(nPoints):
                # Assign each point to a node. A point is assigned to the
                # node that it is closest to.
                # Note: this now means the voronoi bin numbers start from zero
                vBinNum[j] = numpy.argmin((xPoints[j]-xNode)**2 + (yPoints[j]-yNode)**2)
            
            # Compute centroids of these Vorononi Bins. But now using a dens^2
            # weighting. The dens^2 weighting produces equal-mass Voronoi bins.
            # See Capellari and Copin (2003)
            for j in xrange(nNodes):
                indices = numpy.where(vBinNum==j)[0]
                if len(indices) != 0:
                    xBar, yBar = self.weighted_centroid(xPoints[indices], yPoints[indices], densPoints[indices]**2)
                else:
                    # if the Voronoi bin is empty then give (0,0) as its centroid
                    # then we can catch these empty bins later
                    xBar = 0.0
                    yBar = 0.0
                
                xNode[j] = xBar
                yNode[j] = yBar
            
            delta = numpy.sum((xNode-xNodeOld)**2 + (yNode-yNodeOld)**2)
            iters = iters + 1
            
            print "CVT Iteration: %i, Delta %f" % (iters, delta)
            
            if delta == 0.:
                break
        
        print "CVT complete"
        self.xNode = xNode
        self.yNode = yNode
        self.vBinNum = vBinNum
    
    def weighted_centroid(self, x, y, density):
        """
        Compute the density-weighted centroid of one bin. See Equation 4 of
        Cappellari & Copin (2003).
        
        :param x: array of x-axis spatial coordinates
        :param y: array of y-axis spatial coordiantes
        :param density: array containing the weighting values
        
        :return: tuple `(xBar, yBar)`, the weighted centroid
        """
        mass = numpy.sum(density)
        xBar = numpy.sum(x*density)/mass
        yBar = numpy.sum(y*density)/mass
        return (xBar, yBar)
    
    def get_nodes(self):
        """Returns the x and y positions of the Voronoi nodes."""
        return self.xNode, self.yNode
    
    def get_node_membership(self):
        """Returns an array, the length of the input data arrays in
        `tessellate()`, which have indices into the node arrays of `getNodes()`."""
        return self.vBinNum
    
    def plot_nodes(self, plotPath):
        """Plots the points in each bin as a different colour"""
        from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(6,4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(self.xNode, self.yNode, 'ok')
        canvas.print_figure(plotPath)

class DelaunayTessellation(object):
    """Creates a Delaunay triangulation of the given nodes."""
    def __init__(self, xNode, yNode):
        super(DelaunayTessellation, self).__init__()
        self.xNode = xNode
        self.yNode = yNode
        self.triangulation = Triangulation(self.xNode, self.yNode)
        self.memTable = None # membership table. Can be computed with _computeMembershipTable()
    
    def get_triangulation(self):
        """:return: the Triangulation object (from Kern's package)"""
        return self.triangulation
    
    def get_nodes(self):
        """:return: tuple of (x, y), coordinates of tessellation nodes."""
        return (self.xNode, self.yNode)
    
    def get_extreme_nodes(self):
        """Extreme nodes are those on the convex hull of the whole node distribution.
        :return: numpy array (n, 1) listing the node indices that make up the
            convex hull. Ordered counter-clockwise."""
        return self.triangulation.hull
    
    def get_triangles(self):
        """:return: numpy array (nTri by 3) of node indices of each triangle."""
        return self.triangulation.triangle_nodes
    
    def get_adjacency_matrix(self):
        """Gets matrix describing the neighbouring triangles of each triangle.
        
        .. note:: As per Robert Kern's Delaunay package: The value can also be -1 meaning
            that that edge is on the convex hull of
            the points and there is no neighbor on that edge. The values are ordered
            such that triangle_neighbors[tri, i] corresponds with the edge
            *opposite* triangle_nodes[tri, i]. As such, these neighbors are also in
            counter-clockwise order.
        
        :return: numpy array (nTri by 3) indices of triangles that share a
            side with each triangle. Same order as the triangle array.
        """
        return self.triangulation.triangle_neighbors
    
    def get_circumcenters(self):
        """Gets an array of the circumcenter of each Delaunay triangle.
        :return: numpy array (nTri, 2), whose values are the (x,y) location
            of the circumcenter. Ordered like other triangle arrays.
        """
        return self.triangulation.circumcenters
    
    def get_membership_table(self):
        """The membership table gives the indices of all triangles associated
        with a given node.
        :return: A list nNodes long (in same order as other node arrays). Each
            item is another list giving the triangle indices.
        """
        if self.memTable is None:
            self._make_membership_table()
        return self.memTable
    
    def get_number_of_triangles(self):
        """:return: integer number of triangles in Delaunay triangulation."""
        return self.triangulation.triangle_nodes.shape[0]
    
    def compute_triangle_areas(self):
        """Computes the geometric area of each triangle.
        :return: (nTri, 1) numpy array of triangle areas.
        """
        print "Computing triangle areas"
        # Make abbreviated pointers
        tri = self.get_triangles()
        x = self.xNode
        y = self.yNode
        # Compute side lengths
        a = numpy.sqrt((x[tri[:,1]]-x[tri[:,0]])**2. + (y[tri[:,1]]-y[tri[:,0]])**2.) # between verts 1 and 0
        b = numpy.sqrt((x[tri[:,2]]-x[tri[:,1]])**2. + (y[tri[:,2]]-y[tri[:,1]])**2.) # between verts 2 and 1
        c = numpy.sqrt((x[tri[:,2]]-x[tri[:,0]])**2. + (y[tri[:,2]]-y[tri[:,0]])**2.) # between verts 2 and 0
        s = (a+b+c) / 2. # semi-perimeter
        areas = numpy.sqrt(s*(s-a)*(s-b)*(s-c)) # Heron's formula
        return areas
    
    def compute_voronoi_cell_vertices(self):
        """Computes the polygon vertices of the Voronoi cells surrounding each node."""
        # Function to assess the radial location of a point
        # Based on http://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
        ccwDist = lambda xy: (math.atan2(xy[1] - y0, xy[0] - x0) + 2.*math.pi) % 2.*math.pi
        
        xNode, yNode = self.get_nodes()
        memTable = self.get_membership_table()
        circumcenters = self.get_circumcenters()
        extremeNodes = self.get_extreme_nodes()
        nExtremeNodes = len(extremeNodes)
        print extremeNodes
        cells = []
        for i, members in enumerate(memTable):
            # voronoi vertices are the circumcenters of triangles the node belongs to
            voronoiCell = [circumcenters[j] for j in members]
            # if the node is on the convext hull... then add as points the bisections of
            # of the neighbouring hull edges and the node itself.
            if i in extremeNodes:
                print "%i is on hull; there are %i nodes on hull" % (i, nExtremeNodes)
                # find the neighbouring extreme points, the one to the left and right
                # of the point being studied
                k = extremeNodes.index(i) # index into extreme nodes list
                if k > 0:
                    rightNode = extremeNodes[k-1]
                else:
                    rightNode = extremeNodes[nExtremeNodes-1] # take the wrap-around
                if k < nExtremeNodes-1:
                    leftNode = extremeNodes[k+1]
                else:
                    leftNode = extremeNodes[0]
                # Now left and right nodes are indices into the nodes list
                # of the node's hull neighbours
                print "\tneighbours: %i and %i" % (leftNode, rightNode)
                voronoiCell.append(((xNode[rightNode]+xNode[i])/2.,(yNode[rightNode]+yNode[i])/2.))
                voronoiCell.append(((xNode[leftNode]+xNode[i])/2.,(yNode[leftNode]+yNode[i])/2.))
                voronoiCell.append((xNode[i],yNode[i]))
            # Find the geometric centre of the cell
            nP = len(voronoiCell) # number of vertices of the voronoi polygon
            x0 = sum(p[0] for p in voronoiCell) / nP
            y0 = sum(p[1] for p in voronoiCell) / nP
            voronoiCell.sort(key=ccwDist) # ensure the vertices are sorted CCW
            cells.append(voronoiCell)
        
        return cells
    
    def _make_membership_table(self):
        """Computes the membership table of the triangles that each node is
        associated with."""
        xNode, yNode = self.get_nodes()
        nNodes = len(xNode)
        self.memTable = [None]*nNodes
        nTri = self.get_number_of_triangles()
        triangleMatrix = self.get_triangles()
        vertices = range(3)
        for i in xrange(nTri):
            for j in vertices:
                theNode = triangleMatrix[i,j]
                if self.memTable[theNode] == None:
                    self.memTable[theNode] = [i]
                else:
                    # check to see if the triangle is noted for theNode yet
                    triExists = False
                    for k in range(len(self.memTable[theNode])):
                        if i == self.memTable[theNode][k]:
                            triExists = True
                    if triExists == False:
                        self.memTable[theNode].append(i)


class VoronoiDensityEstimator(object):
    """Uses a DelaunayTessellation and point data to set the density of each
    Voronoi cell. This can then be used as the values for the zeroth-order
    Voronoi density field reconstruction.
    """
    def __init__(self):
        super(VoronoiDensityEstimator, self).__init__()


class DelaunayDensityEstimator(object):
    """Uses the DTFE to set the density of each Delaunay node from a
    DelaunayTessellation and the point data. This can then be used as the values
    for the first-order Delaunay density field reconstruction.
    
    :param delaunayTessellation: A `DelaunayTessellation` object, which has the
        delaunay triangulation
    """
    def __init__(self, delaunayTessellation):
        super(DelaunayDensityEstimator, self).__init__()
        self.delaunayTessellation = delaunayTessellation
        self.nodeDensity = None
    
    def estimate_density(self, xRange, yRange, nodeMasses, pixelMask=None):
        """Estimate the density of each node in the delaunay tessellation using
        the DTFE of Schaap (PhD Thesis, Groningen).
        :param xRange: a tuple of (x_min, x_max); TODO make these optional, just
            to scale the pixel mask array
        :param yRange: a tuple of (y_min, y_max)
        :param nodeMasses: an array of the mass of each node, in the same order
            used by the `delaunayTessellation` object
        :param pixelMask: numpy array of same size as the extent of `xRange`
            and `yRange`. A pixel value of 1 is masked, values of 0 are admitted.
            This allows masked areas to be excluded from the density computation.
        """
        xNode, yNode = self.delaunayTessellation.get_nodes()
        nNodes = len(xNode)
        if nNodes != len(nodeMasses):
            print "Warning: nodeMasses has wrong length (%i, should be %i)" % \
                (len(xNode), len(nodeMasses))
        
        #triangles = self.delaunayTessellation.get_triangles()
        #adjMatrix = self.delaunayTessellation.get_adjacency_matrix()
        memTable = self.delaunayTessellation.get_membership_table()
        nTri = self.delaunayTessellation.get_number_of_triangles()
        areas = self.delaunayTessellation.compute_triangle_areas()
        extremeNodes = self.delaunayTessellation.get_extreme_nodes()
        
        # TODO this is where I would compute the masked areas...
        # use the Walking Triangle algorithm
        maskedAreas = numpy.zeros(nTri)
        if pixelMask is not None:
            pass
        
        # Compute the area of triangles contiguous about each node
        contigAreas = numpy.zeros([nNodes])
        for i in xrange(nNodes):
            contigTris = memTable[i] # list of triangle indices associated with node i
            contigAreas[i] = areas[contigTris].sum() - maskedAreas[contigTris].sum()
        
        # Correct the area of contiguous Voronoi regions on the outside of the
        # convex hull.
        # TODO check this
        nExtremeNodes = len(extremeNodes)
        for i, node in enumerate(extremeNodes):
            # find the neighbouring extreme points, the one to the left and right
            # of the point being studied
            if i > 0:
                rightNode = extremeNodes[i-1]
            else:
                rightNode = extremeNodes[nExtremeNodes-1] # take the wrap-around
            if i < nExtremeNodes-1:
                leftNode = extremeNodes[i+1]
            else:
                leftNode = extremeNodes[0]
            # find the angle that they subtend, using The Law of Cosines
            a = math.sqrt((xNode[leftNode]-xNode[node])**2 + (yNode[leftNode]-yNode[node])**2)
            b = math.sqrt((xNode[rightNode]-xNode[node])**2 + (yNode[rightNode]-yNode[node])**2)
            c = math.sqrt((xNode[rightNode]-xNode[leftNode])**2 + (yNode[rightNode]-yNode[node])**2)
            subAngle = math.acos((a**2+b**2-c**2)/(2*a*b))
            # The sub angle should in theory be less than 180 degrees. This code
            # ensures that we have an angle that covers more than 180 degrees.
            extraAngle = 2*math.pi - subAngle
            if extraAngle < math.pi:
                print "Angle error in edge effect correction"
            correctionFactor = extraAngle / subAngle
            # update the contiguous area:
            contigAreas[node] = (1.+correctionFactor)*contigAreas[node]
        
        # Finally compute the density at the site of each node by using
        # eqn 3.36 of Schaap 2007 (pg 69)
        self.nodeDensity = 3. * nodeMasses / contigAreas
        
        # Compute the total tessellated area
        self.totalArea = contigAreas.sum() / 3.
        
        return self.nodeDensity

class FieldRenderer(object):
    """Renders the Delaunay-tessellated field to a pixel-based image."""
    def __init__(self, delaunayTessellation):
        super(FieldRenderer, self).__init__()
        self.delaunayTessellation = delaunayTessellation
    
    def render_zeroth_order_voronoi(self, nodeValues, xRange, yRange, xStep, yStep):
        """Renders a zeroth-order field (a Voronoi tiling).
        :param nodeValues: numpy array (nNodes, 1) of the field values at each node
        :param xRange: tuple of (x_min, x_max)
        :param yRange: tuple of (y_min, y_max)
        :param xStep: scalar, size of pixels along x-axis
        :param yStep: scalar, size of pixels along y-axis
        """
        cellVertices = self.delaunayTessellation.compute_voronoi_cell_vertices()
        
        xScalarRange = xRange[1]-xRange[0]
        yScalarRange = yRange[1]-yRange[0]
        
        nX = int(xScalarRange / xStep)
        nY = int(yScalarRange / yStep)
        
        # Transform the cell vertices in physical units to the pixel units of
        # the rendering space
        pixVertices = [] # same as `cellVertices` but in pixel space
        for i, cell in enumerate(cellVertices):
            pixCell = []
            for j, vertex in enumerate(cell):
                phyX, phyY = vertex
                pixX = phyX / xScalarRange * nX
                pixY = phyY / yScalarRange * nY
                pixCell.append((pixX,pixY))
            pixVertices.append(pixCell)
        
        # Now paint each of these cells with the colour of the nodeValue.
        # Use PIL for this?
        im = Image.new("F", (nX,nY))
        draw = ImageDraw.Draw(im)
        for i, cell in enumerate(pixVertices):
            draw.polygon(cell, fill=nodeValues[i])
        
        # Convert the PIL image to a numpy array
        # http://effbot.org/zone/pil-changes-116.htm
        imArray = numpy.asarray(im) # will be read-only, so make a copy
        imArrayCopy = numpy.array(imArray, copy=True)
        
        return imArrayCopy
    
    def render_first_order_delaunay(self, nodeValues, xRange, yRange, xStep, yStep, defaultValue=numpy.nan):
        """Renders a linearly interpolated Delaunay field.
        :param nodeValues: numpy array (nNodes, 1) of the field values at each node
        :param xRange: tuple of (x_min, x_max)
        :param yRange: tuple of (y_min, y_max)
        :param xStep: scalar, size of pixels along x-axis
        :param yStep: scalar, size of pixels along y-axis
        :param defaultValue: scalar value used outside the tessellation's convex hull
        """
        interp = self.delaunayTessellation.get_triangulation().linear_interpolator(nodeValues, default_value=defaultValue)
        field = self._run_interpolator(interp, xRange, yRange, xStep, yStep)
        return field
    
    def render_nearest_neighbours_delaunay(self, nodeValues, xRange, yRange, xStep, yStep, defaultValue=numpy.nan):
        """docstring for renderNearestNeighboursDelaunay.
        :param nodeValues: numpy array (nNodes, 1) of the field values at each node
        :param xRange: tuple of (x_min, x_max)
        :param yRange: tuple of (y_min, y_max)
        :param xStep: scalar, size of pixels along x-axis
        :param yStep: scalar, size of pixels along y-axis
        :param defaultValue: scalar value used outside the tessellation's convex hull
        """
        interp = self.delaunayTessellation.getTriangulation().nn_interpolator(self, nodeValues, default_value=defaultValue)
        field = self._runInterpolator(interp, xRange, yRange, xStep, yStep)
        return field
    
    def _run_interpolator(self, interp, xRange, yRange, xStep, yStep):
        """Runs Robert Kern's Linear or NN interpolator objects to create a field."""
        nX = int((xRange[1]-xRange[0])/xStep)
        nY = int((yRange[1]-yRange[0])/yStep)
        field = interp[yRange[0]:yRange[1]:complex(0,nY),xRange[0]:xRange[1]:complex(0,nX)]
        return field

def makeRectangularBinnedDensityField(x, y, mass, xRange, yRange, xBinSize, yBinSize):
    """Does rectangular binning on a point distribution. Returns a field with
    pixels in units of sum(mass_points)/area. Each pixel in the field is a bin.
    If you want a bin to occupy more than one pixel, just resize the numpy array.
    
    :param x: numpy array of x point coordinates
    :param y: numpy array of y point coordinates
    :param mass: numpy array of point masses
    :param xRange: tuple of (xmin, xmax); determines range of reconstructed field
    :param yRange: tuple of (ymin, ymax)
    :param xBinSize: scalar, length of bins along x-axis
    :param yBinSize: scalar, length of bins along y-axis
    """
    xGrid = numpy.arange(min(xRange), max(xRange)-xBinSize, xBinSize)
    yGrid = numpy.arange(min(yRange), max(yRange)-yBinSize, yBinSize)
    field = numpy.zeros([len(yGrid),len(xGrid)])
    binArea = xBinSize * yBinSize
    
    # Trim the dataset to ensure it only covers the range of the field reconstruction
    good = numpy.where((x>min(xRange)) & (x<max(xRange))
        & (y>min(yRange)) & (y<max(yRange)))[0]
    x = x[good]
    y = y[good]
    mass = mass[good]
    
    # Then sort the point into increasing x-coordiante
    xsort = numpy.argsort(x)
    x = x[xsort]
    y = y[xsort]
    mass = mass[xsort]
    
    for xi, xg in enumerate(xGrid):
        col = numpy.where((x>xg) & (x<(xg+xBinSize)))[0]
        if len(col) == 0: continue # no points in whole column
        # xcol = x[col]
        ycol = y[col]
        mcol = mass[col]
        for yi, yg in enumerate(yGrid):
            bin = numpy.where((ycol>yg) & (ycol<(yg+yBinSize)))[0]
            if len(bin) == 0: continue # no points in bin
            totalMass = numpy.sum(mcol[bin])
            field[yi,xi] = totalMass
    
    # Make it a density plot by dividing each pixel by the binArea
    field = field / binArea
    
    return field
