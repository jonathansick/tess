import ghostmap
import numpy
import pyfits


def saveFITS(array, path="test.fits"):
    pyfits.writeto(path, array, clobber=True)

def plot_vo(tri, colors=None):
    import matplotlib as mpl
    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
    # from matplotlib.backends.backend_ps import FigureCanvasPS as FigureCanvas
    from matplotlib.figure import Figure
    
    fig = Figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    fig.subplots_adjust(left=0.15, bottom=0.13,wspace=0.25, right=0.95)
    ax = fig.add_subplot(111, aspect='equal')
    
    if colors is None:
        colors = [(0,1,0,0.2)]
    #lc = mpl.collections.LineCollection(numpy.array(
    #    [(tri.circumcenters[i], tri.circumcenters[j])
    #        for i in xrange(len(tri.circumcenters))
    #            for j in tri.triangle_neighbors[i] if j != -1]),
    #    colors=colors)
    lines = [(tri.circumcenters[i], tri.circumcenters[j])
                for i in xrange(len(tri.circumcenters))
                    for j in tri.triangle_neighbors[i] if j != -1]
    lines = numpy.array(lines)
    lc = mpl.collections.LineCollection(lines, colors=colors)
    # ax = pl.gca()
    ax.add_collection(lc)
    # pl.draw()
    # pl.savefig("voronoi")
    ax.plot(tri.x, tri.y, '.k')
    ax.set_xlim(-50,550)
    ax.set_ylim(-50,550)
    canvas.print_figure("voronoi", dpi=300.)

def guassian_point_process(x0, y0, xSigma, ySigma, nPoints):
    """Returns a x and y coordinates of points sampled from a 2D guassian dist."""
    x = numpy.random.normal(loc=x0, scale=xSigma, size=(nPoints,))
    y = numpy.random.normal(loc=y0, scale=ySigma, size=(nPoints,))
    return x, y


xRange = [0,500]
yRange = [0,500]
x, y = guassian_point_process(250, 250, 100, 50, 20000)
inRange = numpy.where((x>0) & (y>0) & (x<xRange[1]) & (y<yRange[1]))[0]
x = x[inRange]
y = y[inRange]
mass = numpy.ones(len(x))

generator = ghostmap.EqualMassGenerator()
generator.generate_nodes(x, y, None, 100) # bin 10 points together
cvt = ghostmap.CVTessellation()
cvt.tessellate(x, y, mass, preGenerator=generator)
binX, binY = cvt.get_nodes()
mass = numpy.ones(len(binX))

tessellation = ghostmap.DelaunayTessellation(binX,binY)
dtfe = ghostmap.DelaunayDensityEstimator(tessellation)
density = dtfe.estimate_density(xRange, yRange, mass)
renderman = ghostmap.FieldRenderer(tessellation)
field = renderman.render_first_order_delaunay(density, xRange, yRange, 1, 1)
saveFITS(field,path="test_delaunay.fits")
triangulation = tessellation.get_triangulation()
plot_vo(triangulation)

zerothField = renderman.render_zeroth_order_voronoi(density, xRange, yRange, 1, 1)
saveFITS(zerothField, path="test_voronoi.fits")

nnField = renderman.render_nearest_neighbours_delaunay(density, xRange, yRange, 1, 1)
saveFITS(zerothField, path="test_nn.fits")


