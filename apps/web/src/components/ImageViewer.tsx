import React, { useRef, useState, useEffect, useCallback } from 'react';

interface ImageViewerProps {
  src: string;
  alt: string;
  onLoad?: (dimensions: { width: number; height: number }) => void;
  className?: string;
  enableZoom?: boolean;
  enablePan?: boolean;
  maxZoom?: number;
  minZoom?: number;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  src,
  alt,
  onLoad,
  className = '',
  enableZoom = true,
  enablePan = true,
  maxZoom = 3,
  minZoom = 0.5
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [imageLoaded, setImageLoaded] = useState(false);

  // Handle image load
  useEffect(() => {
    if (imgRef.current && src) {
      const img = imgRef.current;
      const handleLoad = () => {
        setImageLoaded(true);
        if (onLoad) {
          onLoad({ width: img.naturalWidth, height: img.naturalHeight });
        }
      };

      if (img.complete) {
        handleLoad();
      } else {
        img.addEventListener('load', handleLoad);
        return () => img.removeEventListener('load', handleLoad);
      }
    }
  }, [src, onLoad]);

  // Reset zoom and pan when image changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [src]);

  // Zoom functions
  const handleZoomIn = useCallback(() => {
    setZoom(prev => Math.min(prev * 1.2, maxZoom));
  }, [maxZoom]);

  const handleZoomOut = useCallback(() => {
    setZoom(prev => Math.max(prev / 1.2, minZoom));
  }, [minZoom]);

  const handleZoomReset = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!enableZoom) return;

    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(minZoom, Math.min(maxZoom, prev * delta)));
  }, [enableZoom, maxZoom, minZoom]);

  // Pan functions
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!enablePan || zoom <= 1) return;

    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [enablePan, zoom, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging || !enablePan) return;

    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  }, [isDragging, enablePan, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Touch events for mobile
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (!enablePan || zoom <= 1 || e.touches.length !== 1) return;

    const touch = e.touches[0];
    setIsDragging(true);
    setDragStart({ x: touch.clientX - pan.x, y: touch.clientY - pan.y });
  }, [enablePan, zoom, pan]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!isDragging || !enablePan || e.touches.length !== 1) return;

    e.preventDefault();
    const touch = e.touches[0];
    setPan({
      x: touch.clientX - dragStart.x,
      y: touch.clientY - dragStart.y
    });
  }, [isDragging, enablePan, dragStart]);

  const handleTouchEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'relative',
        overflow: 'hidden',
        cursor: isDragging ? 'grabbing' : zoom > 1 ? 'grab' : 'default',
        userSelect: 'none'
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <img
        ref={imgRef}
        src={src}
        alt={alt}
        style={{
          transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          transformOrigin: 'center center',
          transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          maxWidth: '100%',
          maxHeight: '100%',
          objectFit: 'contain'
        }}
        draggable={false}
      />

      {/* Zoom controls */}
      {enableZoom && imageLoaded && (
        <div style={{
          position: 'absolute',
          top: 10,
          right: 10,
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          background: 'rgba(255, 255, 255, 0.9)',
          borderRadius: 6,
          padding: 4
        }}>
          <button
            onClick={handleZoomIn}
            disabled={zoom >= maxZoom}
            style={{
              padding: '4px 8px',
              border: '1px solid #ccc',
              borderRadius: 3,
              background: 'white',
              cursor: zoom >= maxZoom ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
            title="Zoom In"
          >
            +
          </button>

          <div style={{
            padding: '2px 8px',
            textAlign: 'center',
            fontSize: '12px',
            color: '#666',
            minWidth: '40px'
          }}>
            {Math.round(zoom * 100)}%
          </div>

          <button
            onClick={handleZoomOut}
            disabled={zoom <= minZoom}
            style={{
              padding: '4px 8px',
              border: '1px solid #ccc',
              borderRadius: 3,
              background: 'white',
              cursor: zoom <= minZoom ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
            title="Zoom Out"
          >
            −
          </button>

          <button
            onClick={handleZoomReset}
            style={{
              padding: '4px 8px',
              border: '1px solid #ccc',
              borderRadius: 3,
              background: 'white',
              cursor: 'pointer',
              fontSize: '12px'
            }}
            title="Reset Zoom"
          >
            ↻
          </button>
        </div>
      )}

      {/* Zoom hint */}
      {enableZoom && imageLoaded && zoom === 1 && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          padding: '4px 8px',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          borderRadius: 3,
          fontSize: '12px',
          pointerEvents: 'none'
        }}>
          Use mouse wheel to zoom • Drag to pan when zoomed
        </div>
      )}
    </div>
  );
};
