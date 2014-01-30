"""Tests making voronoi diagrams and voronoi cell polygons."""

import numpy
import math
import pyfits

import matplotlib
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
# from matplotlib.backends.backend_ps import FigureCanvasPS as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.nxutils as nxutils

import Polygon
import Polygon.Utils

from tess.ghostmap import EqualMassGenerator
from tess.delaunay import DelaunayTessellation, DelaunayDensityEstimator, \
        FieldRenderer
from tess.cvtessellation import CVTessellation

from convexhull import convex_hull

def main():
    # test_testLineSegIntersection()
    tri, xRange, yRange = setup()
    # polyPoints = makeInteriorPolygons(tri)
    polyPoints = makeVoronoiCells(tri, xRange, yRange)
    # extPolyPoints = makeExteriorPolygons(tri, xRange, yRange)
    # plotPolygon(tri, polyPoints, polyPoints.keys()[10], xRange, yRange)
    plotAllPolygons(tri, polyPoints, xRange, yRange)

def setup():
    """docstring for setup"""
    nPoints = 500
    xRange = (0,500)
    yRange = (0,500)
    x = numpy.random.normal(loc=250, scale=100, size=(nPoints,))
    y = numpy.random.normal(loc=250, scale=100, size=(nPoints,))
    inRange = numpy.where((x>0) & (y>0) & (x<xRange[1]) & (y<yRange[1]))[0]
    x = x[inRange]
    y = y[inRange]
    # #numpy.random.uniform(1,50,len(x))
    sigma = 500.
    val = 1./numpy.sqrt(2*numpy.pi*sigma**2.)*numpy.exp(-(x-250.)**2./2./sigma**2.)*numpy.exp(-(y-250.)**2./2./sigma**2.)
    mass = numpy.ones([len(x)])

    generator = EqualMassGenerator()
    generator.generateNodes(x, y, None, 10) # bin 10 points together
    cvt = CVTessellation()
    cvt.tessellate(x, y, mass, preGenerator=generator)
    binX, binY = cvt.getNodes()
    mass = numpy.ones([len(binX)])

    tessellation = DelaunayTessellation(binX,binY)
    dtfe = DelaunayDensityEstimator(tessellation)
    density = dtfe.estimateDensity(xRange, yRange, mass)
    renderman = FieldRenderer(tessellation)
    field = renderman.renderFirstOrderDelaunay(density, xRange, yRange, 1, 1)
    # saveFITS(field,path="test_bin.fits")
    triangulation = tessellation.getTriangulation()
    
    return triangulation, xRange, yRange

def makeVoronoiCells(tri, xRange, yRange):
    """docstring for makeAllPolygons"""
    cells = {}
    cells.update(makeInteriorPolygons(tri))
    cells.update(makeExteriorPolygons(tri, xRange, yRange))
    
    # cells = clipVoronoiCells(cells, xRange, yRange) # DEBUG
    return cells

def clipVoronoiCells(cellDict, xRange, yRange):
    """docstring for clipVoronoiCells"""
    bbPolygon = Polygon.Polygon(((xRange[0],yRange[0]), (xRange[0],yRange[1]),
        (xRange[1],yRange[1]), (xRange[1],yRange[0])))
    for i, cell in cellDict.iteritems():
        print i, cell
        extendedCell = Polygon.Polygon(cell)
        clippedVoronoiPolygon = extendedCell & bbPolygon
        cellDict[i] = Polygon.Utils.pointList(clippedVoronoiPolygon)
    return cellDict

def makeInteriorPolygons(tri):
    """Makes voronoi cell polygons on interior nodes."""
    polyPoints = {}
    nNodes = len(tri.x)
    for iNode in xrange(nNodes):
        polygon = []
        # print tri.hull
        # print iNode
        # print iNode in tri.hull
        if iNode in tri.hull: continue
        adjTriangles = getTrianglesForNode(iNode, tri)
        # get all triangles associated with this node
        # for iTri, triangleNode in enumerate(tri.triangle_nodes):
        for iTri in adjTriangles:
            polygon.append(tuple(tri.circumcenters[iTri]))
        # hullU, hullL = hulls(polygon)
        polyPoints[iNode] = convex_hull(polygon)
        # print polyPoints[iNode]
    return polyPoints

def makeExteriorPolygons(tri, xRange, yRange):
    """Makes voronoi cell polygons on nodes that make up the convex hull.
    Polygons are clipped to the bounding box.
    """
    polyPoints = {}
    nNodes = len(tri.x)
    for iNode in tri.hull:
        adjTriangles = getTrianglesForNode(iNode, tri)
        cellVertices = [tri.circumcenters[adjTriangle] for adjTriangle in adjTriangles]
        
        leadingNode = getLeadingNode(iNode, tri)
        pcLead = getCircumcentreOfCommonTriangle(iNode, leadingNode, tri)
        pdLead = makeExteriorPoint(iNode, leadingNode, pcLead, tri, xRange, yRange)
        cellVertices.append(pdLead)
        
        
        trailingNode = getTrailingNode(iNode, tri)
        pcTrail = getCircumcentreOfCommonTriangle(iNode, trailingNode, tri)
        pdTrail = makeExteriorPoint(iNode, trailingNode, pcTrail, tri, xRange, yRange)
        cellVertices.append(pdTrail)
        # cellVertices = convex_hull(cellVertices)
        
        # extendedVoronoiPolygon = Polygon.Polygon((pcLead, pdLead, pdTrail, pcTrail))
        # extendedVoronoiPolygon = Polygon.Polygon(cellVertices)
        extendedVoronoiPolygon = Polygon.Utils.convexHull(Polygon.Polygon(cellVertices))
        bbPolygon = Polygon.Polygon(((xRange[0],yRange[0]), (xRange[0],yRange[1]),
            (xRange[1],yRange[1]), (xRange[1],yRange[0])))
        clippedVoronoiPolygon = extendedVoronoiPolygon & bbPolygon
        polyPoints[iNode] = Polygon.Utils.pointList(extendedVoronoiPolygon)
        # polyPoints[iNode] = Polygon.Utils.pointList(clippedVoronoiPolygon)
    return polyPoints

def getCircumcentreOfCommonTriangle(node, adjNode, tri):
    """docstring for getCircumcentreOfCommonTriangle"""
    # Get triangle common to both nodes
    nodeTriangles = set(getTrianglesForNode(node, tri))
    adjNodeTriangles = set(getTrianglesForNode(adjNode, tri))
    print nodeTriangles
    print adjNodeTriangles
    commonTriangle = list(adjNodeTriangles.intersection(nodeTriangles))[0]
    print commonTriangle
    # Circumcentre of that common triangle
    pc = tri.circumcenters[commonTriangle]
    return pc

def makeExteriorPoint(node, adjNode, pc, tri, xRange, yRange):
    """
    :return: a tuple (xd,yd) of a point exterior to the bounding box, that
        defines a voronoi cell wall between `node` and `adjNode`.
    """
    # Get coordinates of the two points that define convex hull line seg
    p1 = (tri.x[node], tri.y[node])
    p2 = (tri.x[adjNode], tri.y[adjNode])
    # Get slope of that line
    m = (p2[1]-p1[1]) / (p2[0]-p1[0])
    # Get perpendicular slope; that of the voronoi cell edge
    mPerp = -1./m
    print "mPerp func %.2f" % mPerp
    
    ydf = lambda x: mPerp*(x - pc[0]) + pc[1]
    
    # Get the point pi where the voronoi cell line intersects the convex hull line
    xi = (pc[1]-p1[1] + m*p1[0] - mPerp*pc[0]) / (m - mPerp)
    yi = m * (xi - p1[0]) + p1[1]
    
    # Length of line perpendicular to convex hull line segment; defines where (xd,yd) is from (xc,yc)
    D = math.sqrt((xRange[1]-xRange[0])**2. + (yRange[1]-yRange[0])**2.)
    width = max(xRange)-min(xRange)
    
    pHull = []
    for ii in tri.hull:
        pHull.append((tri.x[ii],tri.y[ii]))
    if nxutils.points_inside_poly([pc], pHull)[0]:
        # the circumcenter is inside the convex hull
        inside = True
    else:
        # the circumcentre is outside the convex hull
        inside = False
    
    xdTest = min(xRange) - width
    ydTest = ydf(xdTest)
    testCellLine = (pc, (xdTest,ydTest))
    hullLine = (p1,p2)
    intersects = testLineSegIntersection(testCellLine, hullLine)
    if intersects and inside:
        # the line intersects the convex hull, as it should
        print "Intersects %s and inside %s" % (str(intersects), str(inside))
        xd = xdTest
        yd = ydTest
    else:
        # must use the opposite orientation
        if (intersects == False) & (inside == False):
            print "Intersects %s and inside %s -- not flipping" % (str(intersects), str(inside))
            xd = xdTest
            yd = ydTest
        else:
            print "Intersects %s and inside %s -- flipping" % (str(intersects), str(inside))
            xd = max(xRange) + width
            yd = ydf(xd)
    
    return (xd, yd)

def test_testLineSegIntersection():
    """docstring for test_testLineSegIntersection"""
    line1 = ((1.1,0.1), (1.5,2.0))
    line2 = ((3.0,1.1), (3.2,1.))
    print testLineSegIntersection(line1, line2)

def testLineSegIntersection(line1, line2):
    """A naive test of whether line segments intersect.
    :param seg*: is a tuple of two (x,y) tuples representing the endpoint of the lines.
    :return: True if the line segments intersect."""
    m1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0])
    m2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0])
    
    xi = (line1[0][1] - line2[0][1] - m1*line1[0][0] + m2*line2[0][0]) / (m2 - m1)
    yi = line1[0][1] - m1*line1[0][0] + m1*xi
    
    # print xi,
    # print yi
    # 
    # fig = Figure(figsize=(4,4))
    # canvas = FigureCanvas(fig)
    # fig.subplots_adjust(left=0.15, bottom=0.13,wspace=0.25, right=0.95)
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.plot([line1[0][0],line1[1][0]], [line1[0][1],line1[1][1]], '-b')
    # ax.plot([line2[0][0],line2[1][0]], [line2[0][1],line2[1][1]], '-r')
    # ax.plot([xi],[yi],'ok')
    # canvas.print_figure("intersect_test", dpi=300.)
    
    left1 = min([line1[0][0], line1[1][0]])
    left2 = min([line2[0][0], line2[1][0]])
    right1 = max([line1[0][0], line1[1][0]])
    right2 = max([line2[0][0], line2[1][0]])
    
    lower1 = min([line1[0][1], line1[1][1]])
    lower2 = min([line2[0][1], line2[1][1]])
    upper1 = max([line1[0][1], line1[1][1]])
    upper2 = max([line2[0][1], line2[1][1]])
    
    if (xi>max((left1,left2))) and (xi<min((right1,right2))) and (yi>max((lower1,lower2))) and (yi<min((upper1,upper2))):
        print "Line segs intersect"
        return True
    else:
        print "Line segs do not intersect"
        return False

def getLeadingNode(i, tri):
    """docstring for getLeadingNode"""
    nHullNode = len(tri.hull)
    iHull = tri.hull.index(i)
    jHull = iHull + 1
    if jHull >= nHullNode:
        jHull = 0
    return tri.hull[jHull]

def getTrailingNode(i, tri):
    """docstring for getTrailingNode"""
    nHullNode = len(tri.hull)
    iHull = tri.hull.index(i)
    jHull = iHull - 1
    if jHull < 0:
        jHull = nHullNode-1
    return tri.hull[jHull]

def getTrianglesForNode(i, tri):
    """docstring for getTrianglesForNode"""
    nodeTriangles = []
    for iTri, nodes in enumerate(tri.triangle_nodes):
        if i in nodes:
            nodeTriangles.append(iTri)
    return nodeTriangles

def plotPolygon(tri, polyPoints, nodeID, xRange, yRange):
    """Plots the voronoi polygon around a single node."""
    fig = Figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    fig.subplots_adjust(left=0.15, bottom=0.13,wspace=0.25, right=0.95)
    ax = fig.add_subplot(111, aspect='equal')
    
    ax.plot(tri.x, tri.y, '.k', ms=1)
    ax.plot(tri.x[nodeID], tri.y[nodeID],'.r', ms=2)
    # print polyPoints[nodeID]
    patch = matplotlib.patches.Polygon(polyPoints[nodeID], closed=True, fill=True, lw=1)
    ax.add_patch(patch)
    # ax.plot(tri.x, tri.y, '.k')
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    canvas.print_figure("cell", dpi=300.)

def plotAllPolygons(tri, polyPoints, xRange, yRange):
    """docstring for plotAllPolygons"""
    fig = Figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    fig.subplots_adjust(left=0.15, bottom=0.13,wspace=0.25, right=0.95)
    ax = fig.add_subplot(111, aspect='equal')
    
    ax.plot(tri.x, tri.y, '.k', ms=2)
    for nodeID, points in polyPoints.iteritems():
        patch = matplotlib.patches.Polygon(points, closed=True, fill=False, lw=1)
        ax.add_patch(patch)
    
    ax.plot(tri.x[tri.hull], tri.y[tri.hull], '.r', ms=3)
    for edge in tri.edge_db:
        p1 = (tri.x[edge[0]], tri.y[edge[0]])
        p2 = (tri.x[edge[1]], tri.y[edge[1]])
        patch = matplotlib.patches.Polygon((p1,p2), lw=0.5, color='g', zOrder=1000)
        ax.add_patch(patch)
    
    pHull = []
    for ii in tri.hull:
        pHull.append((tri.x[ii],tri.y[ii]))
    patch = matplotlib.patches.Polygon(pHull, closed=True, fill=False, lw=0.8, color='b')
    ax.add_patch(patch)
    
    # for iNode in tri.hull:
        # print "Lead:"
        # leadingNode = getLeadingNode(iNode, tri)
        # pcLead = getCircumcentreOfCommonTriangle(iNode, leadingNode, tri)
        # print pcLead
        # pdLead = makeExteriorPoint(iNode, leadingNode, pcLead, tri, xRange, yRange)
        # print pdLead
        # 
        # # print "Trail:"
        # # trailingNode = getTrailingNode(iNode, tri)
        # # pcTrail = getCircumcentreOfCommonTriangle(iNode, trailingNode, tri)
        # # pdTrail = makeExteriorPoint(iNode, trailingNode, pcTrail, tri, xRange, yRange)
        # 
        # # Get coordinates of the two points that define convex hull line seg
        # p1 = (tri.x[iNode], tri.y[iNode])
        # p2 = (tri.x[leadingNode], tri.y[leadingNode])
        # # Get slope of that line
        # m = (p2[1]-p1[1]) / (p2[0]-p1[0])
        # # Get perpendicular slope; that of the voronoi cell edge
        # mPerp = -1./m
        # print "mPerp plot %.2f" % mPerp
        # xt = numpy.linspace(min(xRange), max(xRange), 100)
        # yt = mPerp*(xt-pcLead[0]) + pcLead[1]
        # ax.plot(xt, yt, '--m')
        # 
        # # Get the point pi where the voronoi cell line intersects the convex hull line
        # xi = (pcLead[1]-p1[1] + m*p1[0] - mPerp*pcLead[0]) / (m - mPerp)
        # yi = m * (xi - p1[0]) + p1[1]
        # ax.plot([xi], [yi], 'xm', ms=5, zorder=1000)
        # 
        # ax.plot(tri.x[iNode], tri.y[iNode], 'oy', ms=4, zorder=1000)
        # ax.plot([pcLead[0]], [pcLead[1]], 'om', ms=5, zorder=1000)
        # ax.plot([pcLead[0],pdLead[0]], [pcLead[1],pdLead[1]], '-mo')
        # # ax.plot([pcTrail[0]], [pcTrail[1]], '+c', ms=7, zorder=1100)
        # break
    for iNode in xrange(len(tri.x)):
        patch = matplotlib.patches.Polygon(polyPoints[iNode], closed=True, fill=True, lw=0.5)
        ax.add_patch(patch)
    
    width = xRange[1]-xRange[0]
    height = yRange[1]-yRange[0]
    ax.set_xlim(xRange[0]-0.25*width, xRange[1]+0.25*width)
    ax.set_ylim(yRange[0]-0.25*height, yRange[1]+0.25*height)
    canvas.print_figure("cell", dpi=300.)

if __name__ == '__main__':
    main()
