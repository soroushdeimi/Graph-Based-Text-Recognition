# Graph-Based Text Recognition

A graph-based OCR (Optical Character Recognition) approach that recognizes text by comparing letter graph templates to components in a larger text graph.

## Description

This project implements a graph-based text recognition system that approaches OCR from a structural perspective. Instead of dealing with pixel data, we represent letters as graphs where:

- Nodes are key points in the letter with x,y coordinates
- Edges represent connections between these points
- Each node has a weight (default 1000)
- Edge weights are Euclidean distances between nodes

The system first learns a set of letter templates (A-Z), then recognizes text by decomposing a larger graph into connected components and matching each against the known templates.

## Key Algorithms

The core of this project is based on:

1. **Graph Edit Distance**: Calculating the minimum cost to transform one graph into another
2. **Graph Contraction**: Reducing graphs to comparable sizes through edge and vertex merging
3. **Graph Isomorphism Detection**: Finding the optimal mapping between two graphs

## Problem Description

Given:
- A set of letter templates (A-Z) represented as graphs
- A text graph containing multiple connected components

The task is to:
1. Identify connected components in the text graph
2. Match each component to the closest letter template
3. Arrange recognized letters in proper reading order (left-to-right, top-to-bottom)
4. Output the recognized text

## How It Works

The algorithm works as follows:

1. **Learning Phase**: Store graph templates for each letter
2. **Segmentation**: Find connected components in the text graph
3. **Recognition**: For each component:
   - Calculate graph distance to each template
   - Select the template with minimum distance
4. **Ordering**: Sort recognized letters by position (y-coordinate for lines, x-coordinate within lines)
5. **Output**: Print the recognized text line by line

## Implementation Details

- The graph distance calculation uses memoization to avoid redundant computations
- Graph contraction operations include:
  - Edge contraction: Merge two connected vertices
  - Vertex contraction: Merge a vertex with all its neighbors
- The algorithm can handle text with multiple lines

## Usage

```bash
# Compile the code
g++ -std=c++17 -O2 graph_text_recognition.cpp -o text_recognition

# Run with input file
./text_recognition < input.txt
```

## Input Format

The input consists of:

1. Letter template definitions:
```
NEW_GRAPH a
ADD_VERTEX 1 0 0
ADD_VERTEX 2 1 0
ADD_EDGE 1 2
...
```

2. Text recognition command:
```
READ_TEXT
5
ADD_VERTEX 1 10 20
ADD_VERTEX 2 10 25
ADD_EDGE 1 2
...
```

## Example

Input:
```
NEW_GRAPH a
ADD_VERTEX 1 0 0 
ADD_VERTEX 2 4 0
ADD_EDGE 1 2
NEW_GRAPH b
ADD_VERTEX 1 0 0 
ADD_VERTEX 2 1 0
ADD_VERTEX 3 1 1
ADD_EDGE 1 2
ADD_EDGE 2 3
READ_TEXT
11
ADD_VERTEX 1 50 50
ADD_VERTEX 2 50 49
ADD_VERTEX 3 51 49
ADD_EDGE 1 2
ADD_EDGE 2 3
ADD_VERTEX 4 40 30
ADD_VERTEX 5 42 33
ADD_VERTEX 6 30 30
ADD_VERTEX 7 35 30
ADD_EDGE 4 5
ADD_EDGE 6 7
```

Output:
```
b
aa
```

## Performance Considerations

- The graph isomorphism check has factorial complexity in the number of vertices
- Memoization significantly improves performance for repeated subproblems
- The algorithm is optimized for small graphs (letters typically have few vertices)

---

Created by [soroush184](https://github.com/soroush184) 
