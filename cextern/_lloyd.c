
/* lloyd() - Lloy'd centroidal voronoi tessellation algorithm
 * 
 * Arguments
 * nPoints - number of input points
 * x, y - the x and y coordinates of input sites, length nPoints
 * w - the weight, or density, at each input site, length nPoints
 * nNodes - number of nodes
 * nodeX, nodeY - x,y coordinates of initial guesses for Voronoi nodes
 *     length nNodes, will be replaced with new node sites
 * vBinNum - index of each input site in list of nodes
 *
 * Returns values in nodeX, nodeY and vBinNum
*/
#include <stdio.h>
/* #include <assert.h> */
#include <string.h>

/* idx_min_dist - index of node closest to point (x, y)
 *
*/
long idx_min_dist(double x, double y, double *nodeX, double *nodeY, long nNodes) {
    double minDist2 = 1e20;
    double deltX, deltY, d2;
    long minIdx = 0;
    long i;
    for (i = 0; i < nNodes; i++) {
        deltX = x - nodeX[i];
        deltY = y - nodeY[i];
        d2 = deltX*deltX + deltY*deltY;
        if (d2 < minDist2) {
            minIdx = i;
            minDist2 = d2;
        }
    }
    return minIdx;
}

int weighted_centroid(long nPoints, double *x, double *y, double *w,
        long *vBinNum, long nNodes, double *xBar, double *yBar) {
    long i;
    long j;
    double weightSum[nNodes];
    for(i=0; i < nNodes; i++) {
        xBar[i] = 0.;
        yBar[i] = 0.;
        weightSum[i] = 0.;
    }
    for(i=0; i < nPoints; i++) {
        j = vBinNum[i];
        xBar[j] += w[i] * x[i];
        yBar[j] += w[i] * y[i];
        weightSum[j] += w[i];
    }
    for(i=0; i < nNodes; i++) {
        if(weightSum[i] > 0) {
            xBar[i] = xBar[i] / weightSum[i];
            yBar[i] = yBar[i] / weightSum[i];
        } else {
            // Empty bin. Set to zero and flag it later
            xBar[i] = 0.;
            yBar[i] = 0.;
        }
    }
    return 1;
}


int lloyd(long nPoints, double *x, double *y, double *w,
          long nNodes, double *nodeX, double *nodeY,
          long *vBinNum) {
    long j, binID;
    double origNodeX[nNodes];  // nodeX/Y from prev. iteration
    double origNodeY[nNodes];
    long nodeMemberCount[nNodes];
    double delta, dx, dy;  // store squared distance the nodes have moved
    long iters = 0;
    while(1) {
        // Copy the original nodes
        memcpy(origNodeX, nodeX, sizeof(double) * nNodes);
        memcpy(origNodeY, nodeY, sizeof(double) * nNodes);
        /* assert(origNodeX[0] == nodeX[0]); */
        /* printf("%f", nodeX[0]); */
        /* printf("%f", origNodeX[0]); */

        // Zero the node member counts
        for (j=0; j < nNodes; j++) {
            nodeMemberCount[j] = 0;
        }
        // Assign each point to a node. Points are ssigned to the node they are
        // closest to. This defines a set of voronoi bins.
        for(j=0; j < nPoints; j++) {
            binID = idx_min_dist(x[j], y[j], nodeX, nodeY, nNodes);
            nodeMemberCount[binID] += 1;
            vBinNum[j] = binID;
        }

        // Compute centroids of the Voronoi bins.
        weighted_centroid(nPoints, x, y, w, vBinNum, nNodes, nodeX, nodeY);

        // Compute how much each node has moved
        delta = 0.;
        for(j=0; j < nNodes; j++) {
            dx = origNodeX[j] - nodeX[j];
            dy = origNodeY[j] - nodeY[j];
            delta += dx*dx + dy*dy;
        }
        printf("%ld %.2f\n", iters, delta);
        if(delta == 0.) {
            break;
        } else if(iters >= 300) {
            return 0;
        } else{
            iters += 1;
        }
    }
    return 1;
}

