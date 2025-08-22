Social Network Analysis Tutorial
================================

This tutorial demonstrates how to perform comprehensive social network analysis using Groggy, from data preparation to advanced insights.

Introduction
-----------

Social network analysis (SNA) studies relationships between individuals, organizations, or other entities. We'll analyze a fictional social media platform to demonstrate key concepts and techniques.

Setting Up the Dataset
---------------------

Creating Sample Social Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import groggy as gr
    import random
    import numpy as np
    from datetime import datetime, timedelta

    def create_social_network():
    """Create a realistic social media network"""
    
    # User profiles
    users = [
        {"uid": "alice_researcher", "name": "Alice Johnson", "age": 28, "location": "Boston", 
            "interests": ["AI", "research", "books"], "followers": 1250, "join_date": "2019-03-15"},
        {"uid": "bob_developer", "name": "Bob Chen", "age": 32, "location": "San Francisco", 
            "interests": ["programming", "gaming", "coffee"], "followers": 890, "join_date": "2018-07-22"},
        {"uid": "carol_artist", "name": "Carol Williams", "age": 25, "location": "New York", 
            "interests": ["art", "photography", "travel"], "followers": 2100, "join_date": "2020-01-10"},
        {"uid": "david_teacher", "name": "David Rodriguez", "age": 35, "location": "Chicago", 
            "interests": ["education", "music", "cooking"], "followers": 567, "join_date": "2019-11-03"},
        {"uid": "eve_entrepreneur", "name": "Eve Thompson", "age": 29, "location": "Austin", 
            "interests": ["startups", "tech", "fitness"], "followers": 3200, "join_date": "2017-05-18"},
        {"uid": "frank_journalist", "name": "Frank Miller", "age": 38, "location": "Washington DC", 
            "interests": ["journalism", "politics", "history"], "followers": 4500, "join_date": "2016-09-12"},
        {"uid": "grace_designer", "name": "Grace Kim", "age": 27, "location": "Seattle", 
            "interests": ["design", "UX", "sustainability"], "followers": 1800, "join_date": "2020-06-25"},
        {"uid": "henry_scientist", "name": "Henry Brown", "age": 41, "location": "Boston", 
            "interests": ["physics", "research", "hiking"], "followers": 990, "join_date": "2018-12-08"},
        {"uid": "iris_blogger", "name": "Iris Davis", "age": 24, "location": "Los Angeles", 
            "interests": ["lifestyle", "beauty", "travel"], "followers": 5600, "join_date": "2021-02-14"},
        {"uid": "jack_athlete", "name": "Jack Wilson", "age": 26, "location": "Denver", 
            "interests": ["sports", "fitness", "outdoors"], "followers": 2800, "join_date": "2019-08-30"}
    ]
    
    # Social connections with relationship types
    connections = [
        # Professional connections
        {"source": "alice_researcher", "target": "henry_scientist", "type": "colleague", "strength": 0.8, "duration_months": 18},
        {"source": "bob_developer", "target": "grace_designer", "type": "coworker", "strength": 0.9, "duration_months": 24},
        {"source": "eve_entrepreneur", "target": "frank_journalist", "type": "business", "strength": 0.6, "duration_months": 12},
        
        # Friend connections
        {"source": "alice_researcher", "target": "carol_artist", "type": "friend", "strength": 0.7, "duration_months": 36},
        {"source": "bob_developer", "target": "jack_athlete", "type": "friend", "strength": 0.8, "duration_months": 48},
        {"source": "carol_artist", "target": "iris_blogger", "type": "friend", "strength": 0.6, "duration_months": 15},
        {"source": "david_teacher", "target": "henry_scientist", "type": "friend", "strength": 0.5, "duration_months": 60},
        {"source": "grace_designer", "target": "iris_blogger", "type": "friend", "strength": 0.7, "duration_months": 20},
        
        # Interest-based connections
        {"source": "alice_researcher", "target": "bob_developer", "type": "interest", "strength": 0.4, "duration_months": 8},
        {"source": "carol_artist", "target": "grace_designer", "type": "interest", "strength": 0.8, "duration_months": 22},
        {"source": "eve_entrepreneur", "target": "bob_developer", "type": "interest", "strength": 0.5, "duration_months": 14},
        {"source": "frank_journalist", "target": "david_teacher", "type": "interest", "strength": 0.3, "duration_months": 6},
        {"source": "jack_athlete", "target": "iris_blogger", "type": "interest", "strength": 0.4, "duration_months": 10},
        
        # Location-based connections
        {"source": "alice_researcher", "target": "henry_scientist", "type": "location", "strength": 0.3, "duration_months": 24},
        {"source": "eve_entrepreneur", "target": "jack_athlete", "type": "interest", "strength": 0.2, "duration_months": 5},
        
        # Weak connections (follows/mutual interests)
        {"source": "frank_journalist", "target": "iris_blogger", "type": "follower", "strength": 0.2, "duration_months": 3},
        {"source": "david_teacher", "target": "grace_designer", "type": "follower", "strength": 0.1, "duration_months": 2},
        {"source": "henry_scientist", "target": "eve_entrepreneur", "type": "follower", "strength": 0.2, "duration_months": 7},
    ]
    
    # Build the network
    g = gr.Graph(directed=False)  # Undirected for mutual relationships
    
    # Add users as nodes
    g.add_nodes(users)
    
    # Add connections as edges
    
    g.add_edges(connections, node_mapping=g.get_node_mapping(uid_key='uid'))
    
    return g

    # Create the network
    social_net = create_social_network()
    print(f"Social Network Created:")
    print(f"  Users: {social_net.node_count()}")
    print(f"  Connections: {social_net.edge_count()}")
    print(f"  Density: {social_net.density():.3f}")

Basic Network Properties
-----------------------

Network Overview
~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_network_structure(g):
        """Analyze basic network structure"""
        
        print("=== Network Structure Analysis ===")
        
        # Basic metrics
        print(f"Number of users: {g.node_count()}")
        print(f"Number of connections: {g.edge_count()}")
        print(f"Network density: {g.density():.4f}")
        print(f"Is connected: {g.is_connected()}")
        
        # Degree analysis
        degrees = g.degree()
        degree_values = list(degrees.values())
        
        print(f"\nDegree Statistics:")
        print(f"  Average degree: {np.mean(degree_values):.2f}")
        print(f"  Median degree: {np.median(degree_values):.2f}")
        print(f"  Max degree: {max(degree_values)}")
        print(f"  Min degree: {min(degree_values)}")
        
        # Most connected users
        top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\nMost Connected Users:")
        for user, degree in top_connected:
            user_data = g.nodes[user]
            print(f"  {user_data['name']}: {degree} connections")
        
        return {
            'degrees': degrees,
            'avg_degree': np.mean(degree_values),
            'top_connected': top_connected
        }

    network_stats = analyze_network_structure(social_net)


User Demographics Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_demographics(g):
        """Analyze user demographics and their network implications"""
        
        print("\n=== Demographics Analysis ===")
        
        # Get user data as table
        users_table = g.nodes.table()
        
        # Age analysis
        ages = users_table['age']
        print(f"Age Distribution:")
        print(f"  Average age: {ages.mean():.1f} years")
        print(f"  Age range: {ages.min()} - {ages.max()} years")
        print(f"  Standard deviation: {ages.std():.1f} years")
        
        # Location analysis
        locations = users_table['location'].value_counts()
        print(f"\nLocation Distribution:")
        for location, count in locations.items():
            print(f"  {location}: {count} users")
        
        # Followers analysis
        followers = users_table['followers']
        print(f"\nFollower Statistics:")
        print(f"  Average followers: {followers.mean():.0f}")
        print(f"  Median followers: {followers.median():.0f}")
        print(f"  Range: {followers.min()} - {followers.max()}")
        
        # Most followed users
        top_followed = users_table.sort_by('followers', ascending=False).head(3)
        print(f"\nMost Followed Users:")
        for user in top_followed:
            print(f"  {user['name']}: {user['followers']:,} followers")
        
        return users_table

    demo_analysis = analyze_demographics(social_net)

Relationship Analysis
-------------------

Connection Types and Strength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_relationships(g):
        """Analyze relationship types and strengths"""
        
        print("\n=== Relationship Analysis ===")
        
        # Get edge data
        edges_table = g.edges.table()
        
        # Relationship type distribution
        rel_types = edges_table['type'].value_counts()
        print(f"Relationship Types:")
        for rel_type, count in rel_types.items():
            percentage = (count / len(edges_table)) * 100
            print(f"  {rel_type.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Relationship strength analysis
        strengths = edges_table['strength']
        print(f"\nRelationship Strength:")
        print(f"  Average strength: {strengths.mean():.3f}")
        print(f"  Strong relationships (>0.7): {len(edges_table[edges_table['strength'] > 0.7])}")
        print(f"  Weak relationships (<0.3): {len(edges_table[edges_table['strength'] < 0.3])}")
        
        # Duration analysis
        durations = edges_table['duration_months']
        print(f"\nRelationship Duration:")
        print(f"  Average duration: {durations.mean():.1f} months")
        print(f"  Long-term relationships (>24 months): {len(edges_table[edges_table['duration_months'] > 24])}")
        
        # Strongest relationships
        strongest = edges_table.sort_by('strength', ascending=False).head(3)
        print(f"\nStrongest Relationships:")
        for edge in strongest: # need iterator
            user1_name = g.nodes[edge['source']]['name']
            user2_name = g.nodes[edge['target']]['name']
            print(f"  {user1_name} <-> {user2_name}: {edge['strength']:.2f} ({edge['type']})")
        
        return edges_table

        relationship_analysis = analyze_relationships(social_net)

Homophily Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_homophily(g):
        """Analyze homophily - tendency to connect with similar people"""
        
        print("\n=== Homophily Analysis ===")
        
        # Age homophily
        age_diffs = []
        location_matches = 0
        interest_overlaps = []
        total_edges = 0
        
        for edge in g.edges: # need iterator
            source_data = g.nodes[edge.source].item()
            target_data = g.nodes[edge.target].item()
            
            # Age difference
            age_diff = abs(source_data['age'] - target_data['age'])
            age_diffs.append(age_diff)
            
            # Location similarity
            if source_data['location'] == target_data['location']:
                location_matches += 1
            
            # Interest overlap
            source_interests = set(source_data['interests'])
            target_interests = set(target_data['interests'])
            overlap = len(source_interests.intersection(target_interests))
            interest_overlaps.append(overlap)
            
            total_edges += 1
        
        print(f"Age Homophily:")
        print(f"  Average age difference: {np.mean(age_diffs):.1f} years")
        print(f"  Same-age connections (<5 years): {sum(1 for diff in age_diffs if diff < 5)}/{total_edges}")
        
        print(f"\nLocation Homophily:")
        location_percentage = (location_matches / total_edges) * 100
        print(f"  Same-location connections: {location_matches}/{total_edges} ({location_percentage:.1f}%)")
        
        print(f"\nInterest Homophily:")
        print(f"  Average interest overlap: {np.mean(interest_overlaps):.1f} interests")
        print(f"  Connections with shared interests: {sum(1 for overlap in interest_overlaps if overlap > 0)}/{total_edges}")
        
        return {
            'age_diffs': age_diffs,
            'location_matches': location_matches,
            'interest_overlaps': interest_overlaps
        }

    homophily_analysis = analyze_homophily(social_net)

Centrality Analysis - # TODO: Add centrality analysis
-----------------

Influence and Importance
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python 

   def analyze_centrality(g): # TODO: Add centrality module - next release
       """Analyze different types of centrality to identify influential users"""
       
       print("\n=== Centrality Analysis ===")
       
       # Calculate different centrality measures
       degree_centrality = g.centrality.degree(normalized=True)
       pagerank = g.centrality.pagerank()
       betweenness = g.centrality.betweenness(normalized=True)
       
       # Create centrality comparison table
       users_table = g.nodes.table()
       centrality_data = []
       
       for user_id in g.nodes:
           user_data = g.nodes[user_id]
           centrality_data.append({
               'user_id': user_id,
               'name': user_data['name'],
               'degree_centrality': degree_centrality[user_id],
               'pagerank': pagerank[user_id],
               'betweenness': betweenness[user_id],
               'followers': user_data['followers']
           })
       
       # Convert to table and sort
       centrality_table = gr.table(centrality_data)
       
       print("Top 5 Users by Different Centrality Measures:")
       
       # Degree centrality (most connections)
       print(f"\nDegree Centrality (Most Connected):")
       top_degree = centrality_table.sort_values('degree_centrality', ascending=False).head(5)
       for _, user in top_degree.iterrows():
           print(f"  {user['name']}: {user['degree_centrality']:.3f}")
       
       # PageRank (influence through network)
       print(f"\nPageRank (Network Influence):")
       top_pagerank = centrality_table.sort_values('pagerank', ascending=False).head(5)
       for _, user in top_pagerank.iterrows():
           print(f"  {user['name']}: {user['pagerank']:.3f}")
       
       # Betweenness (bridge between communities)
       print(f"\nBetweenness Centrality (Network Bridges):")
       top_betweenness = centrality_table.sort_values('betweenness', ascending=False).head(5)
       for _, user in top_betweenness.iterrows():
           print(f"  {user['name']}: {user['betweenness']:.3f}")
       
       return centrality_table

   centrality_results = analyze_centrality(social_net)

Correlation Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_centrality_correlations(centrality_table): # why are we using pandas here?
       """Analyze correlations between different centrality measures and external factors"""
       
       print("\n=== Centrality Correlations ===")
       
       # Convert to pandas for correlation analysis
       df = centrality_table.to_pandas()
       
       # Calculate correlations
       correlations = df[['degree_centrality', 'pagerank', 'betweenness', 'followers']].corr()
       
       print("Correlation Matrix:")
       print(correlations.round(3))
       
       # Specific insights
       degree_pagerank_corr = correlations.loc['degree_centrality', 'pagerank']
       followers_pagerank_corr = correlations.loc['followers', 'pagerank']
       
       print(f"\nKey Insights:")
       print(f"  Degree-PageRank correlation: {degree_pagerank_corr:.3f}")
       print(f"  Followers-PageRank correlation: {followers_pagerank_corr:.3f}")
       
       if degree_pagerank_corr > 0.7:
           print("  → Strong correlation between connections and influence")
       
       if followers_pagerank_corr > 0.5:
           print("  → External popularity aligns with network influence")
       else:
           print("  → Network influence differs from external popularity")
       
       return correlations

   correlations = analyze_centrality_correlations(centrality_results)

Community Detection - # TODO: Add community detection
-----------------

Finding Social Groups
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python 

   def detect_communities(g): # TODO: Add community module - next release
       """Detect and analyze communities in the social network"""
       
       print("\n=== Community Detection ===")
       
       # Detect communities using Louvain algorithm
       communities = g.community.louvain(resolution=1.0)
       modularity = g.community.modularity(communities)
       
       print(f"Detected {len(communities)} communities")
       print(f"Modularity score: {modularity:.3f}")
       
       # Analyze each community
       community_analysis = []
       
       for i, community in enumerate(communities):
           print(f"\nCommunity {i+1} ({len(community)} members):")
           
           # Get community members' data
           members_data = []
           for user_id in community:
               user_data = g.nodes[user_id]
               members_data.append(user_data)
               print(f"  - {user_data['name']} ({user_data['location']})")
           
           # Analyze community characteristics
           locations = [member['location'] for member in members_data]
           ages = [member['age'] for member in members_data]
           interests = []
           for member in members_data:
               interests.extend(member['interests'])
           
           # Most common characteristics
           from collections import Counter
           location_counts = Counter(locations)
           interest_counts = Counter(interests)
           
           community_info = {
               'id': i,
               'size': len(community),
               'avg_age': np.mean(ages),
               'dominant_location': location_counts.most_common(1)[0] if location_counts else None,
               'common_interests': interest_counts.most_common(3),
               'members': community
           }
           
           community_analysis.append(community_info)
           
           print(f"    Average age: {community_info['avg_age']:.1f}")
           if community_info['dominant_location']:
               print(f"    Dominant location: {community_info['dominant_location'][0]}")
           print(f"    Common interests: {[interest for interest, count in community_info['common_interests']]}")
       
       return communities, community_analysis

   communities, community_info = detect_communities(social_net)

.. Inter-Community Analysis - # TODO: Add inter-community analysis - next release
.. ~~~~~~~~~~~~~~~~~~~~~~

.. .. code-block:: python

..    def analyze_community_connections(g, communities):
..        """Analyze connections between different communities"""
       
..        print("\n=== Inter-Community Analysis ===")
       
..        # Create community membership mapping
..        user_to_community = {}
..        for i, community in enumerate(communities):
..            for user in community:
..                user_to_community[user] = i
       
..        # Analyze edges between communities
..        inter_community_edges = []
..        intra_community_edges = []
       
..        for source, target in g.edges:
..            source_community = user_to_community[source]
..            target_community = user_to_community[target]
           
..            if source_community == target_community:
..                intra_community_edges.append((source, target))
..            else:
..                inter_community_edges.append((source, target, source_community, target_community))
       
..        print(f"Intra-community connections: {len(intra_community_edges)}")
..        print(f"Inter-community connections: {len(inter_community_edges)}")
       
..        # Bridge users (users with many inter-community connections)
..        bridge_scores = {}
..        for source, target, source_comm, target_comm in inter_community_edges:
..            bridge_scores[source] = bridge_scores.get(source, 0) + 1
..            bridge_scores[target] = bridge_scores.get(target, 0) + 1
       
..        if bridge_scores:
..            top_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)[:3]
..            print(f"\nTop Bridge Users (connecting communities):")
..            for user, bridge_count in top_bridges:
..                user_data = g.nodes[user]
..                print(f"  {user_data['name']}: {bridge_count} inter-community connections")
       
..        # Community interaction matrix
..        community_matrix = np.zeros((len(communities), len(communities)))
..        for source, target, source_comm, target_comm in inter_community_edges:
..            community_matrix[source_comm][target_comm] += 1
..            community_matrix[target_comm][source_comm] += 1  # Undirected
       
..        print(f"\nCommunity Interaction Matrix:")
..        for i in range(len(communities)):
..            for j in range(len(communities)):
..                if i != j and community_matrix[i][j] > 0:
..                    print(f"  Community {i+1} <-> Community {j+1}: {int(community_matrix[i][j])} connections")
       
..        return bridge_scores, community_matrix

..    bridge_analysis, comm_matrix = analyze_community_connections(social_net, communities)

Network Evolution Analysis
-------------------------

Temporal Patterns - # needs to implement our history tree
~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_temporal_patterns(g):
       """Analyze how relationships formed over time"""
       
       print("\n=== Temporal Analysis ===")
       
       # Get relationship durations
       edges_table = g.edges.table()
       durations = edges_table['duration_months'].values
       
       # Relationship formation timeline
       formation_timeline = []
       current_date = datetime.now()
       
       for _, edge in edges_table.iterrows():
           formation_date = current_date - timedelta(days=edge['duration_months'] * 30)
           formation_timeline.append({
               'date': formation_date,
               'type': edge['type'],
               'strength': edge['strength'],
               'source': edge['source'],
               'target': edge['target']
           })
       
       # Sort by formation date
       formation_timeline.sort(key=lambda x: x['date'])
       
       print("Relationship Formation Timeline (oldest to newest):")
       for event in formation_timeline[:5]:  # Show first 5
           source_name = g.nodes[event['source']]['name']
           target_name = g.nodes[event['target']]['name']
           print(f"  {event['date'].strftime('%Y-%m')}: {source_name} <-> {target_name} ({event['type']})")
       
       # Analyze relationship type evolution
       type_timeline = {}
       for event in formation_timeline:
           year_month = event['date'].strftime('%Y-%m')
           if year_month not in type_timeline:
               type_timeline[year_month] = {'friend': 0, 'colleague': 0, 'interest': 0, 'other': 0}
           
           rel_type = event['type'] if event['type'] in type_timeline[year_month] else 'other'
           type_timeline[year_month][rel_type] += 1
       
       print(f"\nRelationship Type Evolution:")
       for period, types in list(type_timeline.items())[-6:]:  # Last 6 periods
           total = sum(types.values())
           if total > 0:
               print(f"  {period}: {dict(types)} (total: {total})")
       
       return formation_timeline, type_timeline

   temporal_analysis = analyze_temporal_patterns(social_net)

Influence Propagation
-------------------

Information Flow Modeling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def model_influence_propagation(g, seed_user, topic="new_tech"):
    """Model how influence/information spreads through the network"""
    
    print(f"\n=== Influence Propagation Analysis ===")
    print(f"Modeling spread of '{topic}' starting from {g.nodes[seed_user]['name']}")
    
    # Simple influence model based on relationship strength and user characteristics
    influenced_users = {seed_user}
    influence_probabilities = {}
    
    # Calculate influence probability for each user
    for user_id in g.node_ids:
        user_data = g.nodes[user_id].item()
        base_influence = user_data['followers'] / 100  # Base influence from follower count
        influence_probabilities[user_id] = min(base_influence, 0.8)  # Cap at 80%
    
    # Simulate propagation through network
    propagation_steps = []
    current_wave = {seed_user}
    step = 0
    
    while current_wave and step < 5:  # Limit to 5 steps
        next_wave = set()
        step_influences = []
        
        for influencer in current_wave:
            neighbors = g.neighbors(influencer)
            
            for neighbor in neighbors:
                if neighbor not in influenced_users:
                    # Calculate influence probability
                    edge_data = g.filter_edges(gr.EdgeFilter.connects_nodes(influencer, neighbor)).edges[0]
                    relationship_strength = edge_data['strength'].value
                    base_prob = influence_probabilities[neighbor]
                    
                    # Influence probability based on relationship strength and base probability
                    influence_prob = relationship_strength * base_prob
                    
                    # Simulate influence (using probability)
                    if random.random() < influence_prob:
                        influenced_users.add(neighbor)
                        next_wave.add(neighbor)
                        step_influences.append({
                            'influencer': influencer,
                            'influenced': neighbor,
                            'probability': influence_prob,
                            'relationship': edge_data['type']
                        })
        
        if step_influences:
            propagation_steps.append({
                'step': step + 1,
                'new_influenced': len(next_wave),
                'influences': step_influences
            })
            
            print(f"\nStep {step + 1}: {len(next_wave)} new users influenced")
            for inf in step_influences[:3]:  # Show first 3
                influencer_name = g.nodes[inf['influencer']]['name']
                influenced_name = g.nodes[inf['influenced']]['name']
                print(f"  {influencer_name} → {influenced_name} (p={inf['probability']:.2f}, {inf['relationship']})")
        
        current_wave = next_wave
        step += 1
    
    total_influenced = len(influenced_users)
    coverage = (total_influenced / g.node_count()) * 100
    
    print(f"\nPropagation Results:")
    print(f"  Total users influenced: {total_influenced}/{g.node_count()} ({coverage:.1f}%)")
    print(f"  Propagation steps: {len(propagation_steps)}")
    
    return influenced_users, propagation_steps

    # Test influence propagation from different seed users
    seed_users = ["frank_journalist", "iris_blogger", "eve_entrepreneur"]
    node_mapping = social_net.get_node_mapping(uid_key='uid')
    seed_users = [node_mapping[uid] for uid in seed_users]

    for seed in seed_users:
        propagation_result = model_influence_propagation(social_net, seed)

Network Recommendations
---------------------

Friend Recommendations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def recommend_connections(g, target_user, num_recommendations=3):
       """Recommend new connections for a user based on network structure"""
       
       print(f"\n=== Connection Recommendations for {g.nodes[target_user]['name']} ===")
       
       target_data = g.nodes[target_user]
       current_connections = set(g.neighbors(target_user))
       current_connections.add(target_user)  # Don't recommend self
       
       recommendations = []
       
       # Analyze all potential connections
       for candidate_id in g.nodes:
           if candidate_id in current_connections:
               continue
           
           candidate_data = g.nodes[candidate_id]
           score = 0
           reasons = []
           
           # Common neighbors (friend-of-friend)
           candidate_neighbors = set(g.neighbors(candidate_id))
           common_neighbors = current_connections.intersection(candidate_neighbors)
           if common_neighbors:
               score += len(common_neighbors) * 0.4
               mutual_friends = [g.nodes[neighbor]['name'] for neighbor in common_neighbors if neighbor != target_user]
               if mutual_friends:
                   reasons.append(f"Mutual connections: {', '.join(mutual_friends[:2])}")
           
           # Similar interests
           target_interests = set(target_data['interests'])
           candidate_interests = set(candidate_data['interests'])
           common_interests = target_interests.intersection(candidate_interests)
           if common_interests:
               score += len(common_interests) * 0.3
               reasons.append(f"Shared interests: {', '.join(list(common_interests))}")
           
           # Same location
           if target_data['location'] == candidate_data['location']:
               score += 0.2
               reasons.append(f"Same location: {target_data['location']}")
           
           # Similar age
           age_diff = abs(target_data['age'] - candidate_data['age'])
           if age_diff <= 5:
               score += 0.1
               reasons.append(f"Similar age ({candidate_data['age']} vs {target_data['age']})")
           
           if score > 0:
               recommendations.append({
                   'user_id': candidate_id,
                   'name': candidate_data['name'],
                   'score': score,
                   'reasons': reasons
               })
       
       # Sort by score and return top recommendations
       recommendations.sort(key=lambda x: x['score'], reverse=True)
       
       print(f"Top {num_recommendations} recommendations:")
       for i, rec in enumerate(recommendations[:num_recommendations]):
           print(f"{i+1}. {rec['name']} (score: {rec['score']:.2f})")
           for reason in rec['reasons']:
               print(f"     • {reason}")
       
       return recommendations[:num_recommendations]

   # Generate recommendations for different users
   for user_id in ["alice_researcher", "bob_developer"]:
       recommendations = recommend_connections(social_net, user_id)

Content Recommendations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def recommend_content(g, target_user):
       """Recommend content based on network interests and influence"""
       
       print(f"\n=== Content Recommendations for {g.nodes[target_user]['name']} ===")
       
       target_data = g.nodes[target_user]
       neighbors = g.neighbors(target_user)
       
       # Analyze neighbor interests weighted by relationship strength
       interest_scores = {}
       
       for neighbor in neighbors:
           neighbor_data = g.nodes[neighbor]
           edge_data = g.get_edge(target_user, neighbor)
           relationship_weight = edge_data['strength']
           
           for interest in neighbor_data['interests']:
               if interest not in target_data['interests']:  # New interests only
                   interest_scores[interest] = interest_scores.get(interest, 0) + relationship_weight
       
       # Sort and recommend top interests
       recommended_interests = sorted(interest_scores.items(), key=lambda x: x[1], reverse=True)
       
       print("Recommended content topics:")
       for interest, score in recommended_interests[:5]:
           print(f"  • {interest.title()} (relevance score: {score:.2f})")
           
           # Find which connections are interested in this topic
           interested_connections = []
           for neighbor in neighbors:
               if interest in g.nodes[neighbor]['interests']:
                   interested_connections.append(g.nodes[neighbor]['name'])
           
           if interested_connections:
               print(f"    Interested connections: {', '.join(interested_connections[:3])}")
       
       return recommended_interests

   # Generate content recommendations
   for user_id in ["alice_researcher", "david_teacher"]:
       content_recs = recommend_content(social_net, user_id)

Visualization and Export
-----------------------

Network Visualization # TODO: need to implement our viz module
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def prepare_network_visualization(g, communities):
       """Prepare network data for visualization"""
       
       print("\n=== Preparing Visualization Data ===")
       
       # Create visualization data structure
       viz_data = {
           'nodes': [],
           'edges': [],
           'communities': communities
       }
       
       # Community color mapping
       colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
       
       # Create community membership mapping
       user_to_community = {}
       for i, community in enumerate(communities):
           for user in community:
               user_to_community[user] = i
       
       # Prepare node data
       for user_id in g.nodes:
           user_data = g.nodes[user_id]
           community_id = user_to_community.get(user_id, 0)
           
           viz_data['nodes'].append({
               'id': user_id,
               'name': user_data['name'],
               'age': user_data['age'],
               'location': user_data['location'],
               'followers': user_data['followers'],
               'community': community_id,
               'color': colors[community_id % len(colors)],
               'size': min(user_data['followers'] / 200, 50)  # Scale for visualization
           })
       
       # Prepare edge data
       for source, target in g.edges:
           edge_data = g.get_edge(source, target)
           
           viz_data['edges'].append({
               'source': source,
               'target': target,
               'type': edge_data['type'],
               'strength': edge_data['strength'],
               'width': edge_data['strength'] * 5  # Scale for visualization
           })
       
       return viz_data

   viz_data = prepare_network_visualization(social_net, communities)

Export for External Tools
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def export_for_analysis(g, filename_prefix="social_network"):
       """Export network data for external analysis tools"""
       
       print(f"\n=== Exporting Data ===")
       
       # Export nodes
       nodes_df = g.nodes.table().to_pandas()
       nodes_file = f"{filename_prefix}_nodes.csv"
       nodes_df.to_csv(nodes_file, index=False)
       print(f"Nodes exported to: {nodes_file}")
       
       # Export edges
       edges_df = g.edges.table().to_pandas()
       edges_file = f"{filename_prefix}_edges.csv"
       edges_df.to_csv(edges_file, index=False)
       print(f"Edges exported to: {edges_file}")
       
       # Export centrality measures
       centrality_df = centrality_results.to_pandas()
       centrality_file = f"{filename_prefix}_centrality.csv"
       centrality_df.to_csv(centrality_file, index=False)
       print(f"Centrality measures exported to: {centrality_file}")
       
       # Export for Gephi (GraphML format)
       try:
           nx_graph = g.to_networkx()
           import networkx as nx
           graphml_file = f"{filename_prefix}.graphml"
           nx.write_graphml(nx_graph, graphml_file)
           print(f"GraphML exported to: {graphml_file}")
       except ImportError:
           print("NetworkX not available for GraphML export")
       
       return {
           'nodes_file': nodes_file,
           'edges_file': edges_file,
           'centrality_file': centrality_file
       }

   # Export data (uncomment to actually export)
   # exported_files = export_for_analysis(social_net)

Summary and Insights
-------------------

.. code-block:: python

   def generate_network_report(g, centrality_results, communities):
       """Generate a comprehensive network analysis report"""
       
       print("\n" + "="*50)
       print("SOCIAL NETWORK ANALYSIS REPORT")
       print("="*50)
       
       # Network overview
       print(f"\n📊 NETWORK OVERVIEW")
       print(f"   Total users: {g.node_count()}")
       print(f"   Total connections: {g.edge_count()}")
       print(f"   Network density: {g.density():.4f}")
       print(f"   Communities detected: {len(communities)}")
       
       # Key influencers
       top_pagerank = centrality_results.sort_values('pagerank', ascending=False).head(3)
       print(f"\n🌟 TOP INFLUENCERS (by PageRank)")
       for i, (_, user) in enumerate(top_pagerank.iterrows(), 1):
           print(f"   {i}. {user['name']} (score: {user['pagerank']:.3f})")
       
       # Network bridges
       top_betweenness = centrality_results.sort_values('betweenness', ascending=False).head(3)
       print(f"\n🌉 NETWORK BRIDGES (by Betweenness)")
       for i, (_, user) in enumerate(top_betweenness.iterrows(), 1):
           print(f"   {i}. {user['name']} (score: {user['betweenness']:.3f})")
       
       # Community insights
       print(f"\n👥 COMMUNITY INSIGHTS")
       for i, info in enumerate(community_info):
           print(f"   Community {i+1}: {info['size']} members, avg age {info['avg_age']:.1f}")
           if info['dominant_location']:
               print(f"      Dominant location: {info['dominant_location'][0]}")
       
       # Recommendations
       print(f"\n💡 KEY RECOMMENDATIONS")
       
       # Find users with low centrality but high follower count (underutilized influence)
       df = centrality_results.to_pandas()
       underutilized = df[(df['followers'] > df['followers'].median()) & 
                         (df['pagerank'] < df['pagerank'].median())]
       
       if len(underutilized) > 0:
           print(f"   • Users with high followers but low network influence:")
           for _, user in underutilized.head(2).iterrows():
               print(f"     - {user['name']}: {user['followers']} followers, PageRank {user['pagerank']:.3f}")
       
       # Density recommendation
       if g.density() < 0.1:
           print(f"   • Network is sparse - consider facilitating more connections")
       elif g.density() > 0.5:
           print(f"   • Network is very dense - may benefit from community-based features")
       
       print(f"\n" + "="*50)

   # Generate final report
   generate_network_report(social_net, centrality_results, communities)

This comprehensive tutorial demonstrates how to perform social network analysis with Groggy, from basic network properties to advanced influence modeling and recommendation systems. The techniques shown here can be adapted to analyze real social networks, organizational structures, or any relationship-based data.