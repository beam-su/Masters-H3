const h3 = require('h3-js');
const fs = require('fs');

// Define the bounding box of London (longitude, latitude format)
const london_bbox = [
    [-0.510375, 51.286760],  // Southwest
    [0.334015, 51.286760],   // Southeast
    [0.334015, 51.691874],   // Northeast
    [-0.510375, 51.691874],  // Northwest
    [-0.510375, 51.286760]   // Closing loop
];

// Define H3 resolution
const resolution = 9;

// Generate hexagons inside the polygon
const hexagons = h3.polyfill(london_bbox, resolution, true);

// Convert hexagons to GeoJSON format
const geojson = {
    type: 'FeatureCollection',
    features: hexagons.map(hex => ({
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [h3.h3ToGeoBoundary(hex, true).map(coord => [coord[0], coord[1]])]
        },
        properties: { h3_index: hex }
    }))
};

// Save the GeoJSON file
fs.writeFileSync('london_h3_hexagons.geojson', JSON.stringify(geojson, null, 2));

console.log('GeoJSON file saved: london_h3_hexagons.geojson');
