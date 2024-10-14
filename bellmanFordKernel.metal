#include <metal_stdlib>
using namespace metal;

struct Edge {
    uint src;
    uint dst;
    float weight;
};

kernel void bellmanFord(
    device const Edge* edges [[ buffer(0) ]],
    device const uint* nodeEdgeStart [[ buffer(1) ]],
    device const float* distances [[ buffer(2) ]],
    device atomic_float* newDistances [[ buffer(3) ]],
    constant uint& numNodes [[ buffer(4) ]],
    device atomic_uint* updatedFlag [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= numNodes) return;

    float dist_u = distances[tid];

    // If the distance to node 'u' is INFINITY, skip
    if (dist_u == INFINITY) return;

    // Get the start and end indices for the current node's edges
    //ie, nodeEdgeStart[2] = 3 -> means node 2 ka edge details starts at index 3 in edgeArray,
    //that means go to edgeArray[3] you will find src, dst and weight where src = 2 meaning source node of this specific edge is 2.
    uint start = nodeEdgeStart[tid];
    
    //This index tells the thread where the outgoing edges for the node end in the edges array.
    //and also, if nodeEdgeStart[3] = 6. ie, it gives 5.. that means outgoing edges for node 2 are stored at indices 3 to 5 (that is one less tahn 6) in the edges array, this also means node 2 has 3 outgoing edges, that is why for node 3 the edges starts at index 6 in the edgesArray
    //purpose: The range of indices [start, end) is the set of outgoing edges for the current node tid.
    //By looping through this range, the thread processes all the outgoing edges of the node.
    uint end = nodeEdgeStart[tid + 1];

    // Iterate over all outgoing edges for node 'u'
    for (uint i = start; i < end; i++) {
        Edge edge = edges[i];
        uint v = edge.dst;
        float newDist = dist_u + edge.weight;

        float oldDist = atomic_load_explicit(&newDistances[v], memory_order_relaxed);

        while (newDist < oldDist) {
            //we got the updated shorter path to node v, hence This atomic operation
            //tries to update the value of newDistances[v] from oldDist to newDist.
            //this means exchange succeeded
            bool exchanged = atomic_compare_exchange_weak_explicit(
                &newDistances[v],
                &oldDist,
                newDist,
                memory_order_relaxed,
                memory_order_relaxed
            );
            if (exchanged) {
                //updatedFlag is set to 1 using an atomic store operation
                // Setting the atomic flag to 1 shows an update occurred
                atomic_store_explicit(updatedFlag, 1, memory_order_relaxed);
                break;
            }
            
            if (!(newDist < oldDist)) {
                //If the compare-exchange operation fails (because newDist is not less than oldDist anymore),
                //the loop breaks, and no further updates are attempted for this edge.
                break;
            }
        }
    }
}

