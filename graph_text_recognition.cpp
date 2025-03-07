#include <bits/stdc++.h>
using namespace std;

const double INF = 1e9;

// Graph structure to store letter representation
struct Graph {
    vector<int> ids;
    vector<double> x_coords, y_coords, weights;
    vector<vector<double>> edges;
    unordered_map<int, int> id_map;
    
    int size() const { return ids.size(); }
};

// Calculate Euclidean distance between two points
double dist(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// Add a vertex to the graph
void add_vertex(Graph &g, int id, double x, double y) {
    g.id_map[id] = g.size();
    g.ids.push_back(id);
    g.x_coords.push_back(x);
    g.y_coords.push_back(y);
    g.weights.push_back(1000.0); // Default weight
    
    int n = g.size();
    g.edges.resize(n);
    for (int i = 0; i < n-1; i++)
        g.edges[i].resize(n, -1);
    g.edges[n-1].resize(n, -1);
    g.edges[n-1][n-1] = 0; // Self-loop weight = 0
}

// Add an edge between two vertices
void add_edge(Graph &g, int id1, int id2) {
    // Check if the vertices exist
    if (g.id_map.find(id1) == g.id_map.end() || g.id_map.find(id2) == g.id_map.end())
        return;
        
    int i = g.id_map[id1], j = g.id_map[id2];
    double w = dist(g.x_coords[i], g.y_coords[i], g.x_coords[j], g.y_coords[j]);
    g.edges[i][j] = w;
    g.edges[j][i] = w; // Undirected graph
}

// Check if two graphs can be isomorphic and compute the distance
double isomorphic_dist(const Graph &g1, const Graph &g2) {
    int n = g1.size();
    if (n != g2.size()) // Different sizes, can't be isomorphic
        return INF;

    vector<int> p(n);
    for(int i = 0; i < n; i++) p[i] = i; // Initialize permutation
    
    double min_dist = INF;
    do {
        double cost = 0;
        // Vertex weight differences
        for (int i = 0; i < n; i++)
            cost += fabs(g1.weights[i] - g2.weights[p[i]]);
        
        bool ok = true;
        // Edge structure and weight differences
        for (int i = 0; i < n && ok; i++) {
            for (int j = i+1; j < n; j++) {
                bool e1 = (g1.edges[i][j] >= 0);
                bool e2 = (g2.edges[p[i]][p[j]] >= 0);
                
                // Edge exists in one graph but not the other
                if (e1 != e2) {
                    ok = false;
                    break;
                }
                
                // Both have edge - add weight difference
                if (e1) 
                    cost += fabs(g1.edges[i][j] - g2.edges[p[i]][p[j]]);
            }
        }
        
        if (ok)
            min_dist = min(min_dist, cost);
            
    } while (next_permutation(p.begin(), p.end()));
    
    return min_dist;
}

// Contract an edge (u,v) in graph g
Graph contract_edge(const Graph &g, int u, int v, double &cost) {
    Graph ng; // new graph after contraction
    vector<int> keep;
    
    // Keep all vertices except u and v
    for (int i = 0; i < g.size(); i++) {
        if (i != u && i != v)
            keep.push_back(i);
    }
    
    // Create new vertex from u and v
    double newW = g.weights[u] + g.weights[v] + g.edges[u][v];
    double newX = (g.x_coords[u] + g.x_coords[v]) / 2.0;
    double newY = (g.y_coords[u] + g.y_coords[v]) / 2.0;

    // Add vertices to new graph
    for (int idx : keep) {
        add_vertex(ng, g.ids[idx], g.x_coords[idx], g.y_coords[idx]);
        ng.weights[ng.size()-1] = g.weights[idx]; // Copy weight
    }
    
    // Add merged vertex
    add_vertex(ng, 0, newX, newY); // Use ID 0 for contracted vertex
    ng.weights[ng.size()-1] = newW;

    int new_idx = ng.size() - 1;
    
    // Connect merged vertex to others
    for (int i = 0; i < new_idx; i++) {
        int old_idx = keep[i];
        double w1 = g.edges[u][old_idx];
        double w2 = g.edges[v][old_idx];
        
        // Add edges if either existed before
        if (w1 >= 0 || w2 >= 0) {
            double sum = 0;
            if (w1 >= 0) sum += w1;
            if (w2 >= 0) sum += w2;
            ng.edges[i][new_idx] = sum;
            ng.edges[new_idx][i] = sum;
        }
    }

    // Copy remaining edges
    for (int i = 0; i < new_idx; i++) {
        for (int j = i+1; j < new_idx; j++) {
            int oi = keep[i], oj = keep[j];
            ng.edges[i][j] = g.edges[oi][oj];
            ng.edges[j][i] = g.edges[oj][oi];
        }
    }
    
    cost = newW; // Return total cost of contraction
    return ng;
}

// Contract a vertex and all its neighbors
Graph contract_vertex(const Graph &g, int v, double &cost) {
    // Find all neighbors of v
    vector<int> nbrs;
    for (int i = 0; i < g.size(); i++) {
        if (i != v && g.edges[v][i] >= 0)
            nbrs.push_back(i);
    }
    
    // Can't contract isolated vertex
    if (nbrs.empty()) {
        cost = INF;
        return g;
    }

    // Calculate new vertex properties
    double newW = g.weights[v];
    double sumX = g.x_coords[v], sumY = g.y_coords[v];
    for (int nbr : nbrs) {
        newW += g.weights[nbr] + g.edges[v][nbr]; // Add neighbor weight and edge weight
        sumX += g.x_coords[nbr];
        sumY += g.y_coords[nbr];
    }
    double newX = sumX / (nbrs.size() + 1);
    double newY = sumY / (nbrs.size() + 1);

    // Identify vertices to remove
    unordered_set<int> remove_set;
    remove_set.insert(v);
    for (int nbr : nbrs) {
        remove_set.insert(nbr);
    }

    // Create new graph
    Graph ng;
    vector<int> keep_idx;
    
    // Keep vertices not in the contraction
    for (int i = 0; i < g.size(); i++) {
        if (remove_set.count(i) == 0) {
            add_vertex(ng, g.ids[i], g.x_coords[i], g.y_coords[i]);
            ng.weights[ng.size()-1] = g.weights[i];
            keep_idx.push_back(i);
        }
    }
    
    // Add the new merged vertex
    add_vertex(ng, 0, newX, newY);
    ng.weights[ng.size()-1] = newW;
    int new_idx = ng.size() - 1;

    // Connect remaining vertices to new vertex
    for (int i = 0; i < new_idx; i++) {
        int old_i = keep_idx[i];
        double sum = 0;
        bool has_edge = false;
        
        // Check for edges to any removed vertex
        for (int j : remove_set) {
            if (g.edges[old_i][j] >= 0) {
                sum += g.edges[old_i][j];
                has_edge = true;
            }
        }
        
        if (has_edge) {
            ng.edges[i][new_idx] = sum;
            ng.edges[new_idx][i] = sum;
        }
    }

    // Preserve existing edges between kept vertices
    for (int i = 0; i < new_idx; i++) {
        for (int j = i+1; j < new_idx; j++) {
            int old_i = keep_idx[i];
            int old_j = keep_idx[j];
            ng.edges[i][j] = g.edges[old_i][old_j];
            ng.edges[j][i] = g.edges[old_j][old_i];
        }
    }

    cost = newW;
    return ng;
}

// Create a hash key for a graph (for memoization)
string hash_graph(const Graph &g) {
    string key = to_string(g.size()) + ";";
    
    // Add adjacency matrix
    for (const auto &row : g.edges) {
        for (double w : row) {
            key += to_string(w) + ",";
        }
        key += ";";
    }
    
    // Add vertex weights
    for (double w : g.weights) {
        key += to_string(w) + ",";
    }
    
    return key;
}

// Global memoization cache
unordered_map<string, vector<pair<Graph, double>>> memo;

// Generate all possible ways to contract a graph to a target size
vector<pair<Graph, double>> contract_to_size(const Graph &g, int target) {
    // Generate cache key
    string key = hash_graph(g) + "|" + to_string(target);
    
    // Check cache
    if (memo.count(key))
        return memo[key];
    
    vector<pair<Graph, double>> results;
    
    // Base case: already at target size
    if (g.size() == target) {
        results.emplace_back(g, 0.0);
        return memo[key] = results;
    }
    
    // Can't reach target size
    if (g.size() < target)
        return memo[key] = results;

    // Try contracting edges
    for (int u = 0; u < g.size(); ++u) {
        for (int v = u+1; v < g.size(); ++v) {
            // Skip if no edge exists
            if (g.edges[u][v] < 0)
                continue;
                
            double cost;
            Graph new_g = contract_edge(g, u, v, cost);
            auto subresults = contract_to_size(new_g, target);
            
            for (auto &p : subresults) {
                results.emplace_back(p.first, p.second + cost);
            }
        }
    }

    // Try contracting vertices with neighbors
    for (int v = 0; v < g.size(); ++v) {
        double cost;
        Graph new_g = contract_vertex(g, v, cost);
        
        if (cost >= INF) // Skip if contraction failed
            continue;
            
        auto subresults = contract_to_size(new_g, target);
        for (auto &p : subresults) {
            results.emplace_back(p.first, p.second + cost);
        }
    }

    // Sort by cost
    sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });
    
    // Remove duplicates (keep cheapest)
    if (!results.empty()) {
        vector<pair<Graph, double>> unique_results;
        unique_results.push_back(results[0]);
        
        for (size_t i = 1; i < results.size(); i++) {
            if (hash_graph(results[i].first) != hash_graph(unique_results.back().first)) {
                unique_results.push_back(results[i]);
            }
        }
        results = unique_results;
    }

    return memo[key] = results;
}

// Calculate graph edit distance between two graphs
double graph_distance(const Graph &g1, const Graph &g2) {
    int max_tgt = min(g1.size(), g2.size());
    double best = INF;
    
    // Try contracting both graphs to each possible size
    for (int tgt = 1; tgt <= max_tgt; tgt++) {
        // Get all possible contractions to size tgt
        auto g1_contractions = contract_to_size(g1, tgt);
        auto g2_contractions = contract_to_size(g2, tgt);
        
        // Compare each pair of contractions
        for (auto &p1 : g1_contractions) {
            for (auto &p2 : g2_contractions) {
                double iso_dist = isomorphic_dist(p1.first, p2.first);
                if (iso_dist < INF) {
                    // Total distance = contraction costs + isomorphic distance
                    best = min(best, p1.second + p2.second + iso_dist);
                }
            }
        }
        
        // Optimization: if we found a good match, we can stop
        // (commented out to ensure we find global optimum)
        // if (best < INF / 2) break;
    }
    
    return best;
}

// Find connected components in a graph using DFS
void find_components(const Graph &g, vector<vector<int>> &components) {
    int n = g.size();
    vector<bool> vis(n, false);
    vector<vector<int>> adj_list(n);
    
    // Build adjacency list for faster traversal
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && g.edges[i][j] >= 0) {
                adj_list[i].push_back(j);
            }
        }
    }
    
    // DFS function
    function<void(int, vector<int>&)> dfs = [&](int u, vector<int> &comp) {
        vis[u] = true;
        comp.push_back(u);
        for (int v : adj_list[u]) {
            if (!vis[v]) {
                dfs(v, comp);
            }
        }
    };
    
    // Find all components
    for (int i = 0; i < n; ++i) {
        if (!vis[i]) {
            vector<int> comp;
            dfs(i, comp);
            components.push_back(comp);
        }
    }
}

// Extract a subgraph based on component vertices
Graph extract_subgraph(const Graph &g, const vector<int> &comp) {
    Graph sub;
    
    // Create mapping from original indices to new indices
    for (size_t i = 0; i < comp.size(); ++i) {
        int idx = comp[i];
        add_vertex(sub, g.ids[idx], g.x_coords[idx], g.y_coords[idx]);
        sub.weights.back() = g.weights[idx];
    }
    
    // Copy edges
    for (size_t i = 0; i < comp.size(); ++i) {
        for (size_t j = 0; j < comp.size(); ++j) {
            int oi = comp[i], oj = comp[j];
            sub.edges[i][j] = g.edges[oi][oj];
        }
    }
    
    return sub;
}

// Recognize text from a graph based on letter templates
void recognize_text(const map<char, Graph> &letter_db, const Graph &text_graph) {
    // Step 1: Find connected components (each component is a letter)
    vector<vector<int>> components;
    find_components(text_graph, components);
    
    // Step 2: Recognize each letter and store with position
    vector<tuple<double, double, char>> letters; // (y, x, letter)
    
    for (const auto &comp : components) {
        // Extract subgraph for this component
        Graph letter_graph = extract_subgraph(text_graph, comp);
        
        // Find position data for this component
        double min_y = INF;
        double avg_x = 0;
        
        for (int idx : comp) {
            min_y = min(min_y, text_graph.y_coords[idx]);
            avg_x += text_graph.x_coords[idx];
        }
        avg_x /= comp.size();
        
        // Find best matching letter
        char best_match = '?';
        double best_dist = INF;
        
        for (const auto &[ch, tmpl] : letter_db) {
            double d = graph_distance(letter_graph, tmpl);
            // Take the closest match or in case of tie, the lexicographically smaller letter
            if (d < best_dist || (d == best_dist && ch < best_match)) {
                best_dist = d;
                best_match = ch;
            }
        }
        
        // Store recognized letter with position
        letters.emplace_back(min_y, avg_x, best_match);
    }
    
    // Sort letters by line (y coordinate) and position in line (x coordinate)
    sort(letters.begin(), letters.end(), [](const auto &a, const auto &b) {
        const double EPS = 1e-6;
        // If y coordinates are close, they're on the same line
        if (fabs(get<0>(a) - get<0>(b)) <= EPS) {
            // Sort by x coordinate (left to right)
            return get<1>(a) < get<1>(b);
        }
        // Sort by y coordinate (top to bottom)
        return get<0>(a) < get<0>(b);
    });
    
    // Output the recognized text, line by line
    if (letters.empty()) return;
    
    double curr_y = get<0>(letters[0]);
    string curr_line;
    
    for (const auto &[y, x, ch] : letters) {
        // Check if we're starting a new line
        if (fabs(y - curr_y) > 1e-6) {
            cout << curr_line << endl;
            curr_line = "";
            curr_y = y;
        }
        curr_line += ch;
    }
    
    // Print the last line
    if (!curr_line.empty()) {
        cout << curr_line << endl;
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    
    map<char, Graph> letters;    // Letter templates
    map<string, Graph> all_graphs;  // All graphs by ID
    
    Graph curr_graph;
    string curr_id;

    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        string cmd;
        iss >> cmd;
        
        if (cmd == "NEW_GRAPH") {
            // Save previous graph if it exists
            if (!curr_id.empty()) {
                all_graphs[curr_id] = curr_graph;
                if (curr_id.size() == 1) { // Single char = letter template
                    letters[curr_id[0]] = curr_graph;
                }
            }
            
            // Start new graph
            iss >> curr_id;
            curr_graph = Graph();
        }
        else if (cmd == "ADD_VERTEX") {
            int id;
            double x, y;
            iss >> id >> x >> y;
            add_vertex(curr_graph, id, x, y);
        }
        else if (cmd == "ADD_EDGE") {
            int a, b;
            iss >> a >> b;
            add_edge(curr_graph, a, b);
        }
        else if (cmd == "GRAPH_DISTANCE") {
            string g1_id, g2_id;
            iss >> g1_id >> g2_id;
            
            double d = graph_distance(all_graphs[g1_id], all_graphs[g2_id]);
            if (d >= INF)
                cout << "inf" << endl;
            else
                cout << fixed << setprecision(6) << d << endl;
        }
        else if (cmd == "READ_TEXT") {
            // Save the current graph before processing
            if (!curr_id.empty()) {
                all_graphs[curr_id] = curr_graph;
                if (curr_id.size() == 1) {
                    letters[curr_id[0]] = curr_graph;
                }
            }
            
            // Reset for reading text
            curr_id.clear();
            curr_graph = Graph();

            // Read text graph
            int m;
            cin >> m;
            cin.ignore();
            
            Graph text_graph;
            for (int i = 0; i < m; ++i) {
                string cmd_line;
                getline(cin, cmd_line);
                istringstream cmd_iss(cmd_line);
                
                string text_cmd;
                cmd_iss >> text_cmd;
                
                if (text_cmd == "ADD_VERTEX") {
                    int id;
                    double x, y;
                    cmd_iss >> id >> x >> y;
                    add_vertex(text_graph, id, x, y);
                }
                else if (text_cmd == "ADD_EDGE") {
                    int a, b;
                    cmd_iss >> a >> b;
                    add_edge(text_graph, a, b);
                }
            }
            
            // Process and recognize text
            recognize_text(letters, text_graph);
            
            // Clean up for next test case
            memo.clear();
            letters.clear();
            all_graphs.clear();
        }
    }
    
    // Save the last graph if there is one
    if (!curr_id.empty() && !curr_graph.ids.empty()) {
        all_graphs[curr_id] = curr_graph;
        if (curr_id.size() == 1) {
            letters[curr_id[0]] = curr_graph;
        }
    }

    return 0;
}