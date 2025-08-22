# Data Analysis Workflow with Groggy

This guide demonstrates how to perform comprehensive data analysis using Groggy's unified storage views for real-world scenarios.

## Scenario: Social Network Analysis

Let's analyze a social network dataset with user profiles and interactions.

### 1. Data Loading and Graph Construction

```python
import groggy as gr
import pandas as pd
import numpy as np

# Create a graph for social network analysis
g = gr.Graph(directed=True)

# Sample user data
users = [
    {'id': 'alice', 'age': 28, 'location': 'NYC', 'occupation': 'Engineer', 'followers': 1250},
    {'id': 'bob', 'age': 34, 'location': 'SF', 'occupation': 'Designer', 'followers': 890},
    {'id': 'charlie', 'age': 29, 'location': 'LA', 'occupation': 'Engineer', 'followers': 2100},
    {'id': 'diana', 'age': 25, 'location': 'NYC', 'occupation': 'Marketing', 'followers': 3400},
    {'id': 'eve', 'age': 31, 'location': 'Seattle', 'occupation': 'Engineer', 'followers': 1780},
    {'id': 'frank', 'age': 27, 'location': 'SF', 'occupation': 'Designer', 'followers': 950},
]

# Add users to graph
g.add_nodes(users)

# Sample interaction data
interactions = [
    {'source': 'alice', 'target': 'bob', 'interaction_type': 'follow', 'strength': 0.8, 'timestamp': '2025-01-15'},
    {'source': 'alice', 'target': 'charlie', 'interaction_type': 'message', 'strength': 0.9, 'timestamp': '2025-01-16'},
    {'source': 'bob', 'target': 'diana', 'interaction_type': 'follow', 'strength': 0.7, 'timestamp': '2025-01-14'},
    {'source': 'charlie', 'target': 'eve', 'interaction_type': 'follow', 'strength': 0.6, 'timestamp': '2025-01-13'},
    {'source': 'diana', 'target': 'frank', 'interaction_type': 'message', 'strength': 0.85, 'timestamp': '2025-01-17'},
    {'source': 'eve', 'target': 'alice', 'interaction_type': 'follow', 'strength': 0.75, 'timestamp': '2025-01-18'},
]

# Add interactions to graph
g.add_edges(interactions)

print(f"Social network: {g.node_count()} users, {g.edge_count()} interactions")
```

### 2. Basic Network Statistics

```python
# Convert to table for analysis
users_table = g.nodes.table()
interactions_table = g.edges.table()

# Basic demographics
print("=== User Demographics ===")
print(users_table.describe())

# Age distribution
ages = users_table['age']
print(f"Age statistics:")
print(f"  Mean: {ages.mean():.1f} years")
print(f"  Median: {ages.median():.1f} years")
print(f"  Range: {ages.min()}-{ages.max()} years")

# Location analysis
locations = users_table['location']
location_counts = locations.value_counts()
print(f"Users by location: {location_counts}")

# Occupation breakdown
occupations = users_table['occupation']
occupation_counts = occupations.value_counts()
print(f"Users by occupation: {occupation_counts}")
```

### 3. Advanced Analytics with GROUP BY

```python
# Analyze users by location and occupation
location_analysis = users_table.group_by('location').agg({
    'age': ['mean', 'std', 'count'],
    'followers': ['mean', 'sum', 'max']
})
print("=== Analysis by Location ===")
print(location_analysis)

# Occupation analysis
occupation_analysis = users_table.group_by('occupation').agg({
    'age': ['mean', 'min', 'max'],
    'followers': ['mean', 'std'],
    'location': 'count'  # Count of users per occupation
})
print("=== Analysis by Occupation ===")
print(occupation_analysis)

# Cross-tabulation: Location vs Occupation
location_occupation = users_table.pivot_table(
    index='location', 
    columns='occupation', 
    values='followers', 
    aggfunc='mean'
)
print("=== Average Followers by Location and Occupation ===")
print(location_occupation)
```

### 4. Interaction Analysis

```python
# Interaction strength analysis
interactions_table = g.edges.table()
strengths = interactions_table['strength']

print("=== Interaction Analysis ===")
print(f"Total interactions: {len(interactions_table)}")
print(f"Average interaction strength: {strengths.mean():.3f}")
print(f"Strongest interaction: {strengths.max():.3f}")
print(f"Weakest interaction: {strengths.min():.3f}")

# Interaction types
interaction_types = interactions_table['interaction_type']
type_counts = interaction_types.value_counts()
print(f"Interaction types: {type_counts}")

# Analyze interaction strength by type
strength_by_type = interactions_table.group_by('interaction_type').agg({
    'strength': ['mean', 'std', 'count'],
    'timestamp': 'count'
})
print("=== Interaction Strength by Type ===")
print(strength_by_type)
```

### 5. Graph-Aware Analysis

```python
# Find influential users using neighborhood analysis
influential_users = []

for user_id in ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank']:
    # Get user's neighborhood (who they interact with)
    neighborhood = gr.GraphTable.neighborhood_table(
        g, user_id, ['followers', 'location', 'occupation']
    )
    
    # Calculate influence metrics
    total_neighbor_followers = neighborhood['followers'].sum() if len(neighborhood) > 0 else 0
    unique_locations = len(neighborhood['location'].unique()) if len(neighborhood) > 0 else 0
    unique_occupations = len(neighborhood['occupation'].unique()) if len(neighborhood) > 0 else 0
    
    user_data = g.get_node(user_id)
    influential_users.append({
        'user_id': user_id,
        'own_followers': user_data['followers'],
        'neighbors_count': len(neighborhood),
        'neighbor_followers_total': total_neighbor_followers,
        'location_diversity': unique_locations,
        'occupation_diversity': unique_occupations,
        'influence_score': user_data['followers'] + total_neighbor_followers * 0.1
    })

# Create influence analysis table
influence_table = gr.table(influential_users)
influence_ranked = influence_table.sort_by('influence_score', ascending=False)

print("=== User Influence Ranking ===")
print(influence_ranked.head())
```

### 6. Multi-Table JOIN Operations

```python
# Create additional performance data
performance_data = gr.table({
    'user_id': ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank'],
    'engagement_rate': [0.12, 0.08, 0.15, 0.22, 0.10, 0.09],
    'content_quality_score': [8.5, 7.2, 9.1, 8.8, 7.9, 8.0],
    'activity_level': ['high', 'medium', 'high', 'very_high', 'medium', 'medium']
})

# JOIN user data with performance data
users_with_performance = users_table.join(performance_data, on='user_id', how='inner')

print("=== Users with Performance Metrics ===")
print(users_with_performance.head())

# Analyze performance by demographics
performance_by_location = users_with_performance.group_by('location').agg({
    'engagement_rate': ['mean', 'std'],
    'content_quality_score': ['mean', 'max'],
    'followers': 'mean'
})
print("=== Performance by Location ===")
print(performance_by_location)

# High performers analysis
high_performers = users_with_performance.filter_rows(
    lambda row: row['engagement_rate'] > 0.10 and row['content_quality_score'] > 8.0
)
print("=== High Performers ===")
print(high_performers[['user_id', 'location', 'occupation', 'engagement_rate', 'content_quality_score']])
```

### 7. Network Connectivity Analysis

```python
# Analyze connectivity patterns
print("=== Network Connectivity ===")

# Calculate degree centrality for each user
degree_centrality = {}
for user_id in ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank']:
    in_degree = len([e for e in interactions if e['target'] == user_id])
    out_degree = len([e for e in interactions if e['source'] == user_id])
    degree_centrality[user_id] = {'in_degree': in_degree, 'out_degree': out_degree, 'total_degree': in_degree + out_degree}

centrality_table = gr.table([
    {'user_id': uid, **metrics} for uid, metrics in degree_centrality.items()
])

# Join with user data for comprehensive analysis
users_with_centrality = users_table.join(centrality_table, on='user_id', how='inner')

# Analyze relationship between followers and network position
correlation_followers_degree = np.corrcoef(
    users_with_centrality['followers'].to_numpy(),
    users_with_centrality['total_degree'].to_numpy()
)[0, 1]

print(f"Correlation between followers and network degree: {correlation_followers_degree:.3f}")

# Find users with high network influence but low follower count (potential influencers)
potential_influencers = users_with_centrality.filter_rows(
    lambda row: row['total_degree'] >= 2 and row['followers'] < 1500
)
print("=== Potential Influencers (High Network Degree, Low Followers) ===")
print(potential_influencers[['user_id', 'location', 'followers', 'total_degree']])
```

### 8. Time-Series Analysis

```python
# Analyze interaction patterns over time
interactions_with_dates = interactions_table.copy()

# Convert timestamp strings to datetime (simplified for example)
import datetime
timestamp_data = []
for _, row in interactions_with_dates.iterrows():
    timestamp_data.append({
        'source': row['source'],
        'target': row['target'],
        'interaction_type': row['interaction_type'],
        'strength': row['strength'],
        'date': datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d').date(),
        'day_of_week': datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d').strftime('%A')
    })

time_analysis_table = gr.table(timestamp_data)

# Analyze interactions by day of week
interactions_by_day = time_analysis_table.group_by('day_of_week').agg({
    'interaction_type': 'count',
    'strength': ['mean', 'sum']
})
print("=== Interactions by Day of Week ===")
print(interactions_by_day)

# Analyze interaction types over time
daily_interactions = time_analysis_table.group_by(['date', 'interaction_type']).agg({
    'strength': ['count', 'mean']
})
print("=== Daily Interaction Patterns ===")
print(daily_interactions)
```

### 9. Export and Integration

```python
# Export analysis results for further processing
users_with_performance.to_csv('social_network_analysis.csv')
influence_ranked.to_csv('user_influence_ranking.csv')

# Convert to pandas for advanced plotting
import matplotlib.pyplot as plt

users_df = users_with_performance.to_pandas()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age vs Followers
axes[0, 0].scatter(users_df['age'], users_df['followers'])
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Followers')
axes[0, 0].set_title('Age vs Followers')

# Engagement Rate vs Content Quality
axes[0, 1].scatter(users_df['engagement_rate'], users_df['content_quality_score'])
axes[0, 1].set_xlabel('Engagement Rate')
axes[0, 1].set_ylabel('Content Quality Score')
axes[0, 1].set_title('Engagement vs Quality')

# Followers by Location
location_followers = users_df.groupby('location')['followers'].mean()
axes[1, 0].bar(location_followers.index, location_followers.values)
axes[1, 0].set_xlabel('Location')
axes[1, 0].set_ylabel('Average Followers')
axes[1, 0].set_title('Average Followers by Location')
axes[1, 0].tick_params(axis='x', rotation=45)

# Content Quality by Occupation
occupation_quality = users_df.groupby('occupation')['content_quality_score'].mean()
axes[1, 1].bar(occupation_quality.index, occupation_quality.values)
axes[1, 1].set_xlabel('Occupation')
axes[1, 1].set_ylabel('Average Content Quality')
axes[1, 1].set_title('Content Quality by Occupation')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('social_network_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 10. Advanced Graph Algorithms

```python
# Apply graph algorithms for deeper insights
components = g.connected_components()
print(f"Network has {len(components)} connected components")

# Analyze each component
for i, component in enumerate(components):
    component_table = component.table()  # Get component as table
    print(f"Component {i+1}: {len(component_table)} users")
    
    # Component demographics
    if len(component_table) > 0:
        avg_age = component_table['age'].mean()
        locations = component_table['location'].unique()
        print(f"  Average age: {avg_age:.1f}")
        print(f"  Locations: {list(locations)}")

# Find shortest paths between users
try:
    path_alice_to_diana = g.shortest_path('alice', 'diana')
    print(f"Shortest path from Alice to Diana: {path_alice_to_diana}")
except:
    print("No path found between Alice and Diana")

# Calculate centrality measures
betweenness = g.centrality.betweenness()
pagerank = g.centrality.pagerank()

# Create centrality analysis table
centrality_analysis = gr.table([
    {
        'user_id': user_id,
        'betweenness': betweenness.get(user_id, 0),
        'pagerank': pagerank.get(user_id, 0)
    }
    for user_id in ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank']
])

# Join with user data for comprehensive centrality analysis
full_analysis = users_with_performance.join(centrality_analysis, on='user_id', how='inner')

# Find users with high centrality
high_centrality_users = full_analysis.filter_rows(
    lambda row: row['betweenness'] > 0.1 or row['pagerank'] > 0.2
)

print("=== High Centrality Users ===")
print(high_centrality_users[['user_id', 'occupation', 'followers', 'betweenness', 'pagerank']])
```

## Summary

This workflow demonstrates how Groggy's unified storage views enable seamless transitions between graph topology analysis and tabular data operations. Key benefits:

1. **Unified API**: Switch between graph operations and table analysis without data conversion overhead
2. **Performance**: Native statistical operations computed in Rust for speed
3. **Flexibility**: Support for complex queries, joins, and aggregations
4. **Integration**: Easy export to pandas, CSV, and visualization libraries
5. **Graph-Aware**: Neighborhood analysis and connectivity-based filtering

The combination of graph algorithms and tabular analytics provides powerful insights that would be difficult to achieve with traditional approaches that treat graph and tabular data separately.