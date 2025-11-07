import React, { useState } from 'react';
import type { FeatureCollection, Position } from '../types/geo';

function pathFromRing(ring: Position[], width: number, height: number): string {
  return ring.map((p,i)=>`${i?'L':'M'} ${p[0]*width} ${p[1]*height}`).join(' ') + ' Z';
}

export const Overlay: React.FC<{ fc: FeatureCollection|null; containerWidth?: number; containerHeight?: number }>=({ fc, containerWidth = 1000, containerHeight = 1000 })=>{
  const [selectedRoom, setSelectedRoom] = useState<string|null>(null);

  if (!fc) return null;

  // Use image dimensions from FeatureCollection if available, otherwise use container size
  const width = fc.image?.width || containerWidth;
  const height = fc.image?.height || containerHeight;
  const viewBox = `0 0 ${width} ${height}`;

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'rgb(59, 130, 246)';
    if (confidence >= 0.8) return 'rgb(40, 167, 69)';   // Green
    if (confidence >= 0.6) return 'rgb(255, 193, 7)';   // Yellow
    return 'rgb(220, 53, 69)';                          // Red
  };

  const getConfidenceOpacity = (confidence?: number) => {
    if (!confidence) return 0.9;
    return Math.max(0.6, confidence);
  };

  return (
    <svg
      viewBox={viewBox}
      style={{
        position:'absolute',
        inset:0,
        pointerEvents:'none',
        width: '100%',
        height: '100%'
      }}
    >
      {fc.features.map((f)=> {
        const isSelected = selectedRoom === f.properties.id;
        const confidence = f.properties.confidence || 0.5;

        return (
          <g key={f.properties.id}>
            {/* Room polygons */}
            {f.geometry.coordinates.map((ring, idx)=> (
              <path
                key={idx}
                d={pathFromRing(ring, width, height)}
                fill={isSelected ? "rgba(59, 130, 246, 0.2)" : "rgba(59, 130, 246, 0.1)"}
                stroke={getConfidenceColor(confidence)}
                strokeWidth={isSelected ? 3 : 2}
                strokeDasharray={isSelected ? "none" : "6 4"}
                strokeOpacity={getConfidenceOpacity(confidence)}
                style={{ cursor: 'pointer', pointerEvents: 'auto' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedRoom(isSelected ? null : f.properties.id);
                }}
              />
            ))}

            {/* Room label */}
            {f.properties.bbox_norm && (
              <g>
                {/* Background rectangle for text */}
                <rect
                  x={(f.properties.bbox_norm[0] + f.properties.bbox_norm[2]) / 2 * width - 30}
                  y={(f.properties.bbox_norm[1] + f.properties.bbox_norm[3]) / 2 * height - 12}
                  width={60}
                  height={16}
                  fill="rgba(255, 255, 255, 0.9)"
                  stroke={getConfidenceColor(confidence)}
                  strokeWidth={1}
                  rx={3}
                />

                {/* Room name/id */}
                <text
                  x={(f.properties.bbox_norm[0] + f.properties.bbox_norm[2]) / 2 * width}
                  y={(f.properties.bbox_norm[1] + f.properties.bbox_norm[3]) / 2 * height}
                  fontSize={Math.max(11, width / 60)}
                  fill={getConfidenceColor(confidence)}
                  fontWeight="bold"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedRoom(isSelected ? null : f.properties.id);
                  }}
                >
                  {f.properties.name_hint ?? f.properties.id}
                </text>

                {/* Confidence score (when selected) */}
                {isSelected && (
                  <text
                    x={(f.properties.bbox_norm[0] + f.properties.bbox_norm[2]) / 2 * width}
                    y={(f.properties.bbox_norm[1] + f.properties.bbox_norm[3]) / 2 * height + 18}
                    fontSize={Math.max(9, width / 80)}
                    fill={getConfidenceColor(confidence)}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    style={{ pointerEvents: 'none' }}
                  >
                    {(confidence * 100).toFixed(0)}% confidence
                  </text>
                )}
              </g>
            )}

            {/* Selection indicator */}
            {isSelected && (
              <circle
                cx={(f.properties.bbox_norm[0] + f.properties.bbox_norm[2]) / 2 * width}
                cy={f.properties.bbox_norm[1] * height - 10}
                r={6}
                fill={getConfidenceColor(confidence)}
                stroke="white"
                strokeWidth={2}
                style={{ pointerEvents: 'none' }}
              />
            )}
          </g>
        );
      })}

      {/* Legend */}
      {fc.features.length > 0 && (
        <g transform={`translate(${width - 120}, 20)`}>
          <rect x={0} y={0} width={110} height={80} fill="rgba(255, 255, 255, 0.9)" stroke="#ddd" strokeWidth={1} rx={4} />
          <text x={8} y={16} fontSize={11} fontWeight="bold" fill="#333">Confidence:</text>
          <circle cx={15} cy={32} r={4} fill="rgb(40, 167, 69)" />
          <text x={25} y={36} fontSize={10} fill="#333">High (80-100%)</text>
          <circle cx={15} cy={48} r={4} fill="rgb(255, 193, 7)" />
          <text x={25} y={52} fontSize={10} fill="#333">Medium (60-80%)</text>
          <circle cx={15} cy={64} r={4} fill="rgb(220, 53, 69)" />
          <text x={25} y={68} fontSize={10} fill="#333">Low (<60%)</text>
        </g>
      )}
    </svg>
  );
};

