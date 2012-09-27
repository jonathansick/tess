
/* lloyd() - Lloy'd centroidal voronoi tessellation algorithm
 * 
 * Arguments
 * x, y - the x and y coordinates of input sites, length N_points
 * w - the weight, or density, at each input site, length N_points
 * nodeX, nodeY - x,y coordinates of initial guesses for Voronoi nodes
 *     length N_nodes, will be replaced with new node sites
 * vBinNum - index of each input site in list of nodes
 *
 * Returns values in nodeX, nodeY and vBinNum
*/
int lloyd(double *x, double *y, double *w, double *nodeX, double *nodeY,
        long *vBinNum) {
    return 1;
}
