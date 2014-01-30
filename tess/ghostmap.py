import numpy as np


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
        nBins = np.max(binNums)
        print "Initial number of bins: %i" % nBins
        
        # The first task is to re-order the bin numbering to get rid of holes
        # caused by bad bins.
        newBinNumber = 0 # indexes in the new-bin
        binIndices = [] # an order list whose elements are the tuples containing
                        # indexes to the bins
        
        for i in xrange(nBins):
            j = i + 1 # j is the actual old bin number
            indices = np.where(binNums==j)[0] # elements in bin j
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
        xNode = np.zeros([nNodes], dtype=np.float)
        yNode = np.zeros([nNodes], dtype=np.float)
        for i in xrange(nNodes):
            indices = binIndices[i]
            xNode[i] = np.average(x[indices])
            yNode[i] = np.average(y[indices])
        
        # Now reassign all unbinned pixels to the nearest good bin as
        # judged by the distance to the bin's centroid
        unbinned = np.where(binNums==0)[0]
        numUnbinned = len(unbinned)
        print "Reassigning %i bins" % numUnbinned
        for i in xrange(numUnbinned):
            dists = (x[unbinned[i]] - xNode)**2 + (y[unbinned[i]] - yNode)**2
            k = np.argmin(dists) # get the closest node
            binNums[unbinned[i]] = k+1 # assign pixel to closest node
        
        # All data elements have now been binned. Now compute the centroids for
        # these final bins. These centroids can be used as generators in a
        # tessellation procedure.
        binIndices = [] # reset the bin indices from above
        print "Created %i bins" % np.max(binNums)
        for i in xrange(nNodes):
            indices = np.where(binNums==(i+1))[0]
            # TODO check if the first statement in the if is ever used
            if len(indices) == 0:
                print "No indices for bin %i" % i
                continue
            else:
                binIndices = binIndices + [indices]
        
        # Calculation of centroids
        for i in xrange(nNodes):
            indices = binIndices[i]
            xNode[i] = np.average(x[indices])
            yNode[i] = np.average(y[indices])
        
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
        binNums = np.zeros([nPoints], dtype=np.uint32)
        
        # good contains 1 if the pixel is in a good bin; otherwise its 0
        good = np.zeros([nPoints], dtype=np.uint32)
        
        # start bin accretion from the pixel with the highest SN
        # currentBin is a vector initialized with the first pixel
        currentBin = np.array([signal.argmax()], dtype=np.uint32)
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
                unbinned = np.where(binNums==0)[0]
                m = unbinned.size
                
                # test if there are no pixels left to bin
                if m == 0:
                    break
                
                # find the unbinned pixel closest to the centroid of the current bin
                dist = (xPoints[unbinned]-xBar)**2 + (yPoints[unbinned]-yBar)**2
                k = np.argmin(dist)
                
                # Add the new unbinned[k] pixel to the "next bin"
                nextBin = np.concatenate((currentBin, np.array([unbinned[k]], dtype=np.uint32)))
                
                # Hang onto the "old" measure before computing new one
                oldSN = currentSN
                currentSN = np.sum(signal[nextBin]) / np.sqrt(np.sum(noise[nextBin]**2))
                
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
                xBar = np.average(xPoints[currentBin])
                yBar = np.average(yPoints[currentBin])
            
            # Recompute lists of what has been binned, and not binned
            unbinned = np.where(binNums == 0)[0]
            binned = np.where(binNums != 0)[0]
            if unbinned.size == 0:
                # if there are no more unbinned pixels, just break...
                break
            
            # Compute geometric centroid of all binned pixels
            xBar = np.average(xPoints[binned])
            yBar = np.average(yPoints[binned])

            # Now find the closest unbinned pixel to the centroid of all the
            # binned pixels and start a new bin from there
            dist = (xPoints[unbinned] - xBar)**2 + (yPoints[unbinned] - yBar)**2
            k = np.argmin(dist) # k is index to unbinned pixel closest to the
                                   # centroid of all binned pixels
            # the next bin is initially made of one pixel
            currentBin = np.array([unbinned[k]], dtype=np.uint32)
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
        binNums = np.zeros([nPoints], dtype=np.uint32)
        # good contains 1 if the pixel is in a good bin; otherwise its 0
        good = np.zeros([nPoints], dtype=np.uint32)
        
        # Start bin accretion of the point closest to the center of the data
        # distribution (ie, near centre of image)
        xCentre = (xPoints.max() - xPoints.min()) / 2.
        yCentre = (yPoints.max() - yPoints.min()) / 2.
        centreDist = (xPoints-xCentre)**2 + (yPoints-yCentre)**2
        currentBin = np.array([centreDist.argmin()], dtype=np.uint32)
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
                unbinned = np.where(binNums==0)[0]
                m = unbinned.size
                
                # test if there are no points left to bin
                if m == 0:
                    break
                
                # find the unbinned point closest to the centroid of the current bin
                dist = (xPoints[unbinned]-xBar)**2 + (yPoints[unbinned]-yBar)**2
                k = np.argmin(dist)
                
                # Add the new unbinned[k] pixel to the "next bin"
                nextBin = np.concatenate((currentBin, np.array([unbinned[k]], dtype=np.uint32)))
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
                    xBar = np.average(xPoints[currentBin])
                    yBar = np.average(yPoints[currentBin])
            
            # Finally, recompute lists of what has been binned, and not binned
            unbinned = np.where(binNums == 0)[0]
            binned = np.where(binNums != 0)[0]
            if unbinned.size == 0:
                # if there are no more unbinned pixels, just break
                break
            
            # Compute geometric centroid of all binned pixels
            xBar = np.average(xPoints[binned])
            yBar = np.average(yPoints[binned])
            
            # Find the closest unbinned pixel to the centroid of all the
            # binned pixels and start a new bin from there
            dist = (xPoints[unbinned]-xBar)**2 + (yPoints[unbinned]-yBar)**2
            k = np.argmin(dist)
            # the next bin is initially made of just one pixel
            currentBin = np.array([unbinned[k]], dtype=np.uint32)
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


