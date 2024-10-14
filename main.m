#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Edge structure representation
typedef struct {
    uint src;
    uint dst;
    float weight;
} Edge;

int main(int argc, const char * argv[]) {
    @autoreleasepool {
//Section 1: giving inputs parameters:
        
        // Manually set the source node here
        //This is the source node label -Very imp
        uint sourceNodeLabel = 2;
        //Set to YES for undirected graphs -Very imp
        BOOL isUndirected = NO;
//Section 2: loading data and parsing
                
        //Reading edge list from the input file
        NSString *filePath = @"/Users/shubham.pandey/Documents/High_Performance_Computing/downloads/roadNet-CA.txt";
        NSError *error = nil;
        NSString *fileContents = [NSString stringWithContentsOfFile:filePath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];

        if (error) {
            NSLog(@"Error reading file: %@", error.localizedDescription);
            return -1;
        }

        NSArray *lines = [fileContents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
        // Data Structures:
        // edgeArray: Will store Edge objects.
        // uniqueNodes: Tracks all unique node labels.
        NSMutableArray *edgeArray = [NSMutableArray array];
        NSMutableSet *uniqueNodes = [NSMutableSet set]; // Set to track unique nodes

        // Line Processing
        // Loop to parse the edge list from the input file
        for (NSString *line in lines) {
            if ([line length] == 0 || [line hasPrefix:@"#"]) continue; // Skip comments and empty lines

            // Replace commas with spaces
            NSString *cleanLine = [line stringByReplacingOccurrencesOfString:@"," withString:@" "];

            // Replace multiple spaces with a single space
            NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:@"\\s+"
                                                                                   options:0
                                                                                     error:nil];
            cleanLine = [regex stringByReplacingMatchesInString:cleanLine
                                                        options:0
                                                          range:NSMakeRange(0, [cleanLine length])
                                                   withTemplate:@" "];

            // Trim leading and trailing whitespace
            cleanLine = [cleanLine stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];

            // Split into components
            NSArray *components = [cleanLine componentsSeparatedByString:@" "];

            // Edge Data Extraction (source node, destination node, and weight):
            if ([components count] >= 2) {
                uint srcLabel = [components[0] intValue];
                uint dstLabel = [components[1] intValue];
                // Default weight for unweighted graphs
                float weight = 1.0f;
                // This condition is true only when the graph is weighted
                if ([components count] == 3) {
                    // Use the weight provided (3rd component: index 2)
                    weight = [components[2] floatValue];
                }

                // Add src and dst to the set of unique nodes
                [uniqueNodes addObject:@(srcLabel)];
                [uniqueNodes addObject:@(dstLabel)];
            } else {
                NSLog(@"Skipping invalid line (fewer than 2 components): %@", line);
            }
        }
//Section 3: Node Label Mapping

        // Mapping node labels to indices (as key-value pairs in dictionary)
        // Acts as a lookup table to map node labels
        NSMutableDictionary *nodeLabelToIndex = [NSMutableDictionary dictionary];
        // Acts as a reverse lookup table to map indices back to node labels
        NSMutableArray *indexToNodeLabel = [NSMutableArray array];

        // Sort node labels to ensure consistent mapping
        NSArray *sortedNodeLabels = [[uniqueNodes allObjects] sortedArrayUsingSelector:@selector(compare:)];

        uint index = 0;
        for (NSNumber *nodeLabel in sortedNodeLabels) {
            [nodeLabelToIndex setObject:@(index) forKey:nodeLabel];
            [indexToNodeLabel addObject:nodeLabel];
            index++;
        }

        uint numNodes = (uint)[uniqueNodes count];
        NSLog(@"Number of Nodes: %u", numNodes);

        // Re-parse the edge list to create Edge structures with indices
        for (NSString *line in lines) {
            if ([line length] == 0 || [line hasPrefix:@"#"]) continue; // Skip comments and empty lines

            // Replace commas with spaces
            NSString *cleanLine = [line stringByReplacingOccurrencesOfString:@"," withString:@" "];

            // Replace multiple spaces with a single space
            NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:@"\\s+"
                                                                                   options:0
                                                                                     error:nil];
            cleanLine = [regex stringByReplacingMatchesInString:cleanLine
                                                        options:0
                                                          range:NSMakeRange(0, [cleanLine length])
                                                   withTemplate:@" "];

            // Trim leading and trailing whitespace
            cleanLine = [cleanLine stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];

            NSArray *components = [cleanLine componentsSeparatedByString:@" "];

            if ([components count] >= 2) {
                uint srcLabel = [components[0] intValue];
                uint dstLabel = [components[1] intValue];
                // Default weight for unweighted graphs
                float weight = 1.0f;

                if ([components count] == 3) {
                    // Use provided weight if present
                    weight = [components[2] floatValue];
                }

                // Map node labels to indices
                uint srcIndex = [[nodeLabelToIndex objectForKey:@(srcLabel)] unsignedIntValue];
                uint dstIndex = [[nodeLabelToIndex objectForKey:@(dstLabel)] unsignedIntValue];

                // Create edge and add to the array
                Edge edge = {srcIndex, dstIndex, weight};
                [edgeArray addObject:[NSValue valueWithBytes:&edge objCType:@encode(Edge)]];

                // For undirected graphs, add the reverse edge
                if (isUndirected) {
                    Edge reverseEdge = {dstIndex, srcIndex, weight};
                    [edgeArray addObject:[NSValue valueWithBytes:&reverseEdge objCType:@encode(Edge)]];
                }
            }
        }

        uint numEdges = (uint)[edgeArray count];

        if (numEdges == 0 || numNodes == 0) {
            NSLog(@"Error: No edges or nodes were loaded. Please check your edge list file.");
            return -1;
        }

        NSLog(@"Number of Edges: %u", numEdges);

        // Modification 1: Sort edges by source node
        //this is an anonymous block/closure that iterates through the edgeAarray elements,
        //comparing pairs of elements to set them into order.
        //For each pair, it calls the comparator block, passing the two elements as arguments obj1 and obj2
        //why? bcz, all OUTGOING edges for a node will be contiguous in memory.
        [edgeArray sortUsingComparator:^NSComparisonResult(NSValue *obj1, NSValue *obj2) {
            Edge edge1, edge2;
            [obj1 getValue:&edge1];
            [obj2 getValue:&edge2];
            if (edge1.src < edge2.src) return NSOrderedAscending;
            if (edge1.src > edge2.src) return NSOrderedDescending;
            return NSOrderedSame;
        }];

        // Modification 2: Build nodeEdgeStart array
        uint *nodeEdgeStart = (uint *)malloc(sizeof(uint) * (numNodes + 1));
        memset(nodeEdgeStart, 0, sizeof(uint) * (numNodes + 1));

        // Count outgoing edges for each node
        for (uint i = 0; i < numEdges; i++) {
            Edge edge;
            [[edgeArray objectAtIndex:i] getValue:&edge];
            nodeEdgeStart[edge.src + 1]++;
        }

        // Compute cumulative sum to get starting indices
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

        // Map the source node label to its index
        NSNumber *sourceNodeKey = @(sourceNodeLabel);
        if (![nodeLabelToIndex objectForKey:sourceNodeKey]) {
            NSLog(@"Error: Source node label %u not found in the graph.", sourceNodeLabel);
            return -1;
        }
        uint sourceNodeIndex = [[nodeLabelToIndex objectForKey:sourceNodeKey] unsignedIntValue];

        distances[sourceNodeIndex] = 0.0f;
        newDistances[sourceNodeIndex] = 0.0f;

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

        // Modification 3 - Prepare buffers
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

        // For newDistanceBuffer, ensure proper alignment for atomic operations
        NSUInteger alignment = sizeof(float); // Alignment for atomic_float
        NSUInteger dataSize = sizeof(float) * numNodes;
        NSUInteger adjustedSize = ((dataSize + alignment - 1) / alignment) * alignment;

        id<MTLBuffer> newDistanceBuffer = [device newBufferWithLength:adjustedSize
                                                              options:MTLResourceStorageModeShared];
        memcpy([newDistanceBuffer contents], newDistances, dataSize);

        // Initialize newDistances to the current distances
        memcpy([newDistanceBuffer contents], [distanceBuffer contents], sizeof(float) * numNodes);

        // Number of nodes buffer
        id<MTLBuffer> numNodesBuffer = [device newBufferWithBytes:&numNodes
                                                           length:sizeof(uint)
                                                          options:MTLResourceStorageModeShared];

        // Create updatedFlag buffer
        uint updatedFlagValue = 0;
        id<MTLBuffer> updatedFlagBuffer = [device newBufferWithBytes:&updatedFlagValue
                                                             length:sizeof(uint)
                                                            options:MTLResourceStorageModeShared];

        // Capture start time
        NSDate *startTime = [NSDate date];

        // Modification 4: Bellman-Ford iterations with updated compute encoder
        for (uint i = 0; i < numNodes - 1; i++) {
            // Reset updatedFlag
            uint *updatedFlagPointer = (uint *)[updatedFlagBuffer contents];
            *updatedFlagPointer = 0;

            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

            [computeEncoder setComputePipelineState:computePipelineState];
            [computeEncoder setBuffer:edgeBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:nodeEdgeStartBuffer offset:0 atIndex:1];
            // Input distances
            [computeEncoder setBuffer:distanceBuffer offset:0 atIndex:2];
            // Output distances (atomic_float)
            [computeEncoder setBuffer:newDistanceBuffer offset:0 atIndex:3];
            [computeEncoder setBuffer:numNodesBuffer offset:0 atIndex:4];
            // Updated flag buffer
            [computeEncoder setBuffer:updatedFlagBuffer offset:0 atIndex:5];

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

            // Check if any updates occurred
            uint updatedFlagValue = *((uint *)[updatedFlagBuffer contents]);
            if (updatedFlagValue == 0) {
                // No updates occurred, break out of the loop
                NSLog(@"No updates in iteration %u. Exiting early.", i + 1);
                break;
            }

            // Swap distance buffers
            id<MTLBuffer> tempBuffer = distanceBuffer;
            distanceBuffer = newDistanceBuffer;
            newDistanceBuffer = tempBuffer;
        }

        // Record end time
        NSDate *endTime = [NSDate date];
        NSTimeInterval executionTime = [endTime timeIntervalSinceDate:startTime];

        // Print the execution time
        NSLog(@"Execution time: %f seconds", executionTime);

        // Read and print the distances on the console
        float *finalDistances = (float *)[distanceBuffer contents];
        for (uint i = 0; i < numNodes; i++) {
            NSNumber *nodeLabel = [indexToNodeLabel objectAtIndex:i];
            if (finalDistances[i] == INFINITY) {
                NSLog(@"Distance from node %u to node %u is INFINITY", sourceNodeLabel, [nodeLabel unsignedIntValue]);
            } else {
                NSLog(@"Distance from node %u to node %u is %f", sourceNodeLabel, [nodeLabel unsignedIntValue], finalDistances[i]);
            }
        }

        // Free allocated memory
        free(distances);
        free(newDistances);
        free(nodeEdgeStart);
    }
    return 0;
}
