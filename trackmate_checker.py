import xml.etree.ElementTree as ET
import pandas as pd

tree = ET.parse('/kaggle/working/sanity_test_tracks.xml')
root = tree.getroot()

# 1. Extract all detected spots
spots = []
for spot in root.findall('.//AllSpots/SpotsInFrame/Spot'):
    spots.append({
        'spot_id': int(spot.attrib['ID']),
        'frame': int(spot.attrib['FRAME']),
        'x': float(spot.attrib['POSITION_X']),
        'y': float(spot.attrib['POSITION_Y']),
        'z': float(spot.attrib['POSITION_Z']),
        'quality': float(spot.attrib['QUALITY']),
    })
df_spots = pd.DataFrame(spots)
print(df_spots.head())

# 2. Extract track edges (parent -> child links)
edges = []
for track in root.findall('.//AllTracks/Track'):
    track_id = int(track.attrib['TRACK_ID'])
    for edge in track.findall('Edge'):
        edges.append({
            'track_id': track_id,
            'source_id': int(edge.attrib['SPOT_SOURCE_ID']),
            'target_id': int(edge.attrib['SPOT_TARGET_ID']),
        })
df_edges = pd.DataFrame(edges)
print(df_edges.head())