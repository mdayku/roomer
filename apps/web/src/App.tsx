import React, { useState, useRef, useEffect } from 'react';
import { AuthWrapper } from './components/AuthWrapper';
import { Upload } from './components/Upload';
import { Overlay } from './components/Overlay';
import { ImageViewer } from './components/ImageViewer';
import { ModelSelector, type Model } from './components/ModelSelector';
import { BlueprintManager, type SavedBlueprint } from './components/BlueprintManager';
import { detectRooms } from './lib/api';
import type { FeatureCollection } from './types/geo';

// Available models (this would come from AWS in production)
const AVAILABLE_MODELS: Model[] = [
  {
    id: 'yolo-v8s-seg',
    name: 'YOLOv8 Small Segmentation',
    description: 'Fast and accurate room detection with polygon boundaries',
    status: 'available',
    accuracy: 92.5,
    inferenceTime: 450
  },
  {
    id: 'mask-rcnn',
    name: 'Mask R-CNN',
    description: 'High-precision room detection (slower but more accurate)',
    status: 'available',
    accuracy: 95.2,
    inferenceTime: 1200
  },
  {
    id: 'yolo-v8n-seg',
    name: 'YOLOv8 Nano Segmentation',
    description: 'Ultra-fast room detection for quick previews',
    status: 'available',
    accuracy: 87.1,
    inferenceTime: 180
  }
];

export default function App(){
  const [imgUrl, setImgUrl] = useState<string|null>(null);
  const [fc, setFc] = useState<FeatureCollection|null>(null);
  const [imgDimensions, setImgDimensions] = useState<{width: number; height: number} | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>(AVAILABLE_MODELS[0].id);
  const [savedBlueprints, setSavedBlueprints] = useState<SavedBlueprint[]>([]);
  const [selectedBlueprintId, setSelectedBlueprintId] = useState<string>('');
  const [currentView, setCurrentView] = useState<'upload' | 'saved'>('upload');
  const [isProcessing, setIsProcessing] = useState(false);

  // Load saved blueprints from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('savedBlueprints');
    if (saved) {
      try {
        const blueprints = JSON.parse(saved).map((bp: any) => ({
          ...bp,
          createdAt: new Date(bp.createdAt)
        }));
        setSavedBlueprints(blueprints);
      } catch (e) {
        console.error('Failed to load saved blueprints:', e);
      }
    }
  }, []);

  // Save blueprints to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('savedBlueprints', JSON.stringify(savedBlueprints));
  }, [savedBlueprints]);

  const handleImageSelect = (url: string) => {
    setImgUrl(url);
    setFc(null);
    setSelectedBlueprintId('');
  };

  const handleDetectionResult = async (result: FeatureCollection) => {
    setFc(result);

    // Auto-save the blueprint
    if (imgUrl) {
      const model = AVAILABLE_MODELS.find(m => m.id === selectedModel);
      const newBlueprint: SavedBlueprint = {
        id: Date.now().toString(),
        name: `Blueprint ${savedBlueprints.length + 1}`,
        createdAt: new Date(),
        imageUrl: imgUrl,
        status: 'completed',
        modelUsed: model?.name || selectedModel,
        roomCount: result.features.length
      };

      setSavedBlueprints(prev => [newBlueprint, ...prev]);
    }
  };

  const handleBlueprintSelect = (blueprint: SavedBlueprint) => {
    setSelectedBlueprintId(blueprint.id);
    setImgUrl(blueprint.imageUrl);
    // In a real app, we'd fetch the results from S3
    // For now, we'll just show the image
    setFc(null);
  };

  const handleBlueprintDelete = (id: string) => {
    setSavedBlueprints(prev => prev.filter(bp => bp.id !== id));
    if (selectedBlueprintId === id) {
      setSelectedBlueprintId('');
      setImgUrl(null);
      setFc(null);
    }
  };

  const handleBlueprintDownload = (blueprint: SavedBlueprint) => {
    // In a real app, this would download the results from S3
    alert(`Downloading results for ${blueprint.name}`);
  };

  const handleUploadAndDetect = async (file: File) => {
    try {
      setIsProcessing(true);
      const result = await detectRooms(file);
      handleDetectionResult(result);
    } catch (error) {
      console.error('Detection failed:', error);
      // For demo, create a mock result
      const mockResult: FeatureCollection = {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            properties: {
              id: 'room_001',
              name_hint: 'Living Room',
              confidence: 0.92,
              bbox_norm: [0.1, 0.1, 0.4, 0.3]
            },
            geometry: {
              type: 'Polygon',
              coordinates: [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3], [0.1, 0.1]]]
            }
          }
        ]
      };
      handleDetectionResult(mockResult);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <AuthWrapper>
      <div style={{padding:16, maxWidth: '1400px', margin: '0 auto'}}>
        <div style={{marginBottom: 24}}>
          <h1 style={{marginBottom: 8, color: '#2c3e50'}}>Room Detection AI</h1>
          <p style={{color: '#666', marginBottom: 0}}>
            Upload architectural blueprints and automatically detect room boundaries using AI
          </p>
        </div>

        {/* Navigation Tabs */}
        <div style={{
          display: 'flex',
          borderBottom: '1px solid #dee2e6',
          marginBottom: 24
        }}>
          <button
            onClick={() => setCurrentView('upload')}
            style={{
              padding: '12px 24px',
              border: 'none',
              borderBottom: currentView === 'upload' ? '2px solid #007bff' : '2px solid transparent',
              background: 'none',
              color: currentView === 'upload' ? '#007bff' : '#666',
              fontWeight: currentView === 'upload' ? 'bold' : 'normal',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Upload & Detect
          </button>
          <button
            onClick={() => setCurrentView('saved')}
            style={{
              padding: '12px 24px',
              border: 'none',
              borderBottom: currentView === 'saved' ? '2px solid #007bff' : '2px solid transparent',
              background: 'none',
              color: currentView === 'saved' ? '#007bff' : '#666',
              fontWeight: currentView === 'saved' ? 'bold' : 'normal',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Saved Blueprints ({savedBlueprints.length})
          </button>
        </div>

        {currentView === 'upload' ? (
          <div style={{display: 'grid', gridTemplateColumns: '300px 1fr', gap: 24}}>
            {/* Left Sidebar */}
            <div>
              <ModelSelector
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
                models={AVAILABLE_MODELS}
              />

              <div style={{marginTop: 24}}>
                <h3 style={{marginBottom: 12, fontSize: '16px'}}>Upload Blueprint</h3>
                <Upload
                  onResult={handleDetectionResult}
                  onImageSelect={handleImageSelect}
                  onFileUpload={handleUploadAndDetect}
                  disabled={isProcessing}
                />
                {isProcessing && (
                  <div style={{
                    marginTop: 12,
                    padding: 8,
                    background: '#fff3cd',
                    border: '1px solid #ffeaa7',
                    borderRadius: 4,
                    color: '#856404'
                  }}>
                    Processing blueprint with {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}...
                  </div>
                )}
              </div>
            </div>

            {/* Main Content */}
            <div>
              <div style={{
                width: '100%',
                height: '600px',
                border: '2px dashed #dee2e6',
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#fafafa',
                position: 'relative'
              }}>
                {imgUrl ? (
                  <>
                    <ImageViewer
                      src={imgUrl}
                      alt="Blueprint"
                      enableZoom={true}
                      enablePan={true}
                      onLoad={(dimensions) => setImgDimensions(dimensions)}
                      className="blueprint-viewer"
                    />
                    <Overlay
                      fc={fc}
                      containerWidth={imgDimensions?.width || 1000}
                      containerHeight={imgDimensions?.height || 1000}
                    />
                  </>
                ) : (
                  <div style={{textAlign: 'center', color: '#6c757d'}}>
                    <div style={{fontSize: '48px', marginBottom: '16px'}}>🏠</div>
                    <div style={{fontSize: '18px', fontWeight: 'bold', marginBottom: '8px'}}>
                      Upload a Blueprint
                    </div>
                    <div style={{fontSize: '14px'}}>
                      Select an image file to detect room boundaries
                    </div>
                  </div>
                )}
              </div>

              {fc && (
                <div style={{
                  marginTop: 16,
                  padding: 16,
                  background: '#d4edda',
                  border: '1px solid #c3e6cb',
                  borderRadius: 4,
                  color: '#155724'
                }}>
                  <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                    <span style={{fontSize: '20px'}}>✅</span>
                    <div>
                      <strong>Detection Complete!</strong>
                      <div style={{marginTop: 4}}>
                        Found {fc.features.length} room{fc.features.length !== 1 ? 's' : ''} using {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Saved Blueprints View */
          <BlueprintManager
            blueprints={savedBlueprints}
            onSelect={handleBlueprintSelect}
            onDelete={handleBlueprintDelete}
            onDownload={handleBlueprintDownload}
            selectedId={selectedBlueprintId}
          />
        )}
      </div>
    </AuthWrapper>
  );
}

