#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

//edge structure representation
typedef struct {
    uint src;
    uint dst;
    float weight;
} Edge;

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        //MANUALLY set the source node here VERY imp.
        uint sourceNode = 3858241;
        //MANUALLY set NO for directed graphs, YES for undirected graph...VERY imp dont forget.
        BOOL isUndirected = NO;

        //read edge list from the input file
        //update the file path
        NSString *filePath = @"/Users/shubham.pandey/Documents/High_Performance_Computing/downloads/cit-Patents.txt";
        NSError *error = nil;
        NSString *fileContents = [NSString stringWithContentsOfFile:filePath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];

        if (error) {
            NSLog(@"Error reading file: %@", error.localizedDescription);
            return -1;
        }

        NSArray *lines = [fileContents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
        NSMutableArray *edgeArray = [NSMutableArray array];

        uint maxNode = 0;

        //parsing the edge list
        for (NSString *line in lines) {
            //to skip comments and empty lines
            if ([line length] == 0 || [line hasPrefix:@"#"]) continue;

            NSArray *components = [line componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];

            if ([components count] >= 2) {
                uint src = [components[0] intValue];
                uint dst = [components[1] intValue];
                //imp to handle default weight for unweighted graphs, take all edge weight as 1
                float weight = 1.0f;
                
                
                if ([components count] == 3) {
                    //means there exists a third column that indicates weghts of corresponding edge in my input edge list file,
                    //-usko weight assume kar ke update kar do
                    weight = [components[2] floatValue];
                }

                Edge edge = {src, dst, weight};
                [edgeArray addObject:[NSValue valueWithBytes:&edge objCType:@encode(Edge)]];

                if (isUndirected) {
                    Edge reverseEdge = {dst, src, weight};
                    [edgeArray addObject:[NSValue valueWithBytes:&reverseEdge objCType:@encode(Edge)]];
                }

                if (src > maxNode) maxNode = src;
                if (dst > maxNode) maxNode = dst;
            } else {
                NSLog(@"Skipping invalid line (fewer than 2 components): %@", line);
            }
        }

        uint numNodes = maxNode + 1;
        uint numEdges = (uint)[edgeArray count];

        if (numEdges == 0 || numNodes == 0) {
            NSLog(@"Error: No edges or nodes were loaded. Please check your edge list file.");
            return -1;
        }

        NSLog(@"Number of Edges: %u", numEdges);
        NSLog(@"Number of Nodes: %u", numNodes);

        //modification 1: Sort edges by destination node
        //Purpose: Organizes edges so that all edges for a particular node are contiguous. This optimizes memory access patterns on the GPU.
        [edgeArray sortUsingComparator:^NSComparisonResult(NSValue *obj1, NSValue *obj2) {
            Edge edge1, edge2;
            [obj1 getValue:&edge1];
            [obj2 getValue:&edge2];
            if (edge1.dst < edge2.dst) return NSOrderedAscending;
            if (edge1.dst > edge2.dst) return NSOrderedDescending;
            return NSOrderedSame;
        }];

        //mdification 2: Build nodeEdgeStart array
        //Purpose is to indicates the starting index of edges for each node in the edges array. This allows quick access to all edges of a node.
        uint *nodeEdgeStart = (uint *)malloc(sizeof(uint) * (numNodes + 1));
        memset(nodeEdgeStart, 0, sizeof(uint) * (numNodes + 1));

        for (uint i = 0; i < numEdges; i++) {
            Edge edge;
            [[edgeArray objectAtIndex:i] getValue:&edge];
            nodeEdgeStart[edge.dst + 1]++;
        }

        //Convert counts to starting indices
        for (uint i = 1; i <= numNodes; i++) {
            nodeEdgeStart[i] += nodeEdgeStart[i - 1];
        }

        // Initialize distances
        float *distances = (float *)malloc(sizeof(float) * numNodes);
        float *newDistances = (float *)malloc(sizeof(float) * numNodes);

        for (uint i = 0; i < numNodes; i++) {
            distances[i] = INFINITY;
            newDistances[i] = INFINITY;
        }
        distances[sourceNode] = 0.0f;
        newDistances[sourceNode] = 0.0f;

        // Create Metal device and command queue
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // Load the Metal shader
        NSError *libraryError = nil;
        id<MTLLibrary> library = [device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&libraryError];

        if (!library) {
            NSLog(@"Error occurred when creating library: %@", libraryError);
            return -1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"bellmanFord"];
        NSError *computePipelineError = nil;
        id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:function error:&computePipelineError];

        if (!computePipelineState) {
            NSLog(@"Error occurred when creating compute pipeline state: %@", computePipelineError);
            return -1;
        }

        // **Modification 3: Prepare buffers**
        // Edge buffer
        id<MTLBuffer> edgeBuffer = [device newBufferWithLength:sizeof(Edge) * numEdges options:MTLResourceStorageModeShared];
        Edge *edgeBufferPointer = (Edge *)[edgeBuffer contents];

        for (uint i = 0; i < numEdges; i++) {
            Edge edge;
            [[edgeArray objectAtIndex:i] getValue:&edge];
            edgeBufferPointer[i] = edge;
        }

        // Node edge start buffer
        id<MTLBuffer> nodeEdgeStartBuffer = [device newBufferWithBytes:nodeEdgeStart
                                                                length:sizeof(uint) * (numNodes + 1)
                                                               options:MTLResourceStorageModeShared];

        // Distance buffers
        id<MTLBuffer> distanceBuffer = [device newBufferWithBytes:distances
                                                           length:sizeof(float) * numNodes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> newDistanceBuffer = [device newBufferWithBytes:newDistances
                                                               length:sizeof(float) * numNodes
                                                              options:MTLResourceStorageModeShared];

        // Initialize newDistances to the current distances
        memcpy([newDistanceBuffer contents], [distanceBuffer contents], sizeof(float) * numNodes);

        // Number of nodes buffer
        id<MTLBuffer> numNodesBuffer = [device newBufferWithBytes:&numNodes
                                                           length:sizeof(uint)
                                                          options:MTLResourceStorageModeShared];

        // Capture start time
        NSDate *startTime = [NSDate date];

        // **Modification 4: Bellman-Ford iterations with updated compute encoder**
        for (uint i = 0; i < numNodes - 1; i++) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

            [computeEncoder setComputePipelineState:computePipelineState];
            [computeEncoder setBuffer:edgeBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:nodeEdgeStartBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:distanceBuffer offset:0 atIndex:2];     // Input distances
            [computeEncoder setBuffer:newDistanceBuffer offset:0 atIndex:3];  // Output distances
            [computeEncoder setBuffer:numNodesBuffer offset:0 atIndex:4];

            MTLSize gridSize = MTLSizeMake(numNodes, 1, 1);
            NSUInteger threadGroupSize = computePipelineState.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numNodes) {
                threadGroupSize = numNodes;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Copy updated distances back to distanceBuffer
            memcpy([distanceBuffer contents], [newDistanceBuffer contents], sizeof(float) * numNodes);
        }

        // Capture end time
        NSDate *endTime = [NSDate date];
        NSTimeInterval executionTime = [endTime timeIntervalSinceDate:startTime];

        // Log the execution time
        NSLog(@"Execution time: %f seconds", executionTime);

        // Read back the distances
        float *finalDistances = (float *)[distanceBuffer contents];
        for (uint i = 0; i < numNodes; i++) {
            if (finalDistances[i] == INFINITY) {
                NSLog(@"Distance from node %u to node %u is INFINITY", sourceNode, i);
            } else {
                NSLog(@"Distance from node %u to node %u is %f", sourceNode, i, finalDistances[i]);
            }
        }

        // Free allocated memory
        free(distances);
        free(newDistances);
        free(nodeEdgeStart);
    }
    return 0;
}
