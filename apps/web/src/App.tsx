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
  const [processingProgress, setProcessingProgress] = useState<string>('');

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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'u':
            e.preventDefault();
            setCurrentView('upload');
            break;
          case 's':
            e.preventDefault();
            setCurrentView('saved');
            break;
          case 'n':
            e.preventDefault();
            if (currentView === 'upload') {
              const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
              fileInput?.click();
            }
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentView]);

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
      setProcessingProgress('Uploading blueprint...');

      // Simulate progress updates (in real app, this would come from the API)
      setTimeout(() => setProcessingProgress('Analyzing blueprint...'), 500);
      setTimeout(() => setProcessingProgress('Detecting rooms...'), 1500);
      setTimeout(() => setProcessingProgress('Processing results...'), 2500);

      const result = await detectRooms(file);
      handleDetectionResult(result);
    } catch (error) {
      console.error('Detection failed:', error);

      // Show user-friendly error message
      alert(`Detection failed: ${error instanceof Error ? error.message : 'Unknown error'}. Using demo results.`);

      // For demo, create a mock result with more rooms
      const mockResult: FeatureCollection = {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            properties: {
              id: 'room_001',
              name_hint: 'Entry Hall',
              confidence: 0.95,
              bbox_norm: [0.05, 0.06, 0.18, 0.24]
            },
            geometry: {
              type: 'Polygon',
              coordinates: [[[0.05, 0.06], [0.18, 0.06], [0.18, 0.24], [0.05, 0.24], [0.05, 0.06]]]
            }
          },
          {
            type: 'Feature',
            properties: {
              id: 'room_002',
              name_hint: 'Living Room',
              confidence: 0.92,
              bbox_norm: [0.22, 0.06, 0.68, 0.45]
            },
            geometry: {
              type: 'Polygon',
              coordinates: [[[0.22, 0.06], [0.68, 0.06], [0.68, 0.28], [0.54, 0.28], [0.54, 0.45], [0.22, 0.45], [0.22, 0.06]]]
            }
          },
          {
            type: 'Feature',
            properties: {
              id: 'room_003',
              name_hint: 'Kitchen',
              confidence: 0.88,
              bbox_norm: [0.72, 0.06, 0.95, 0.35]
            },
            geometry: {
              type: 'Polygon',
              coordinates: [[[0.72, 0.06], [0.95, 0.06], [0.95, 0.35], [0.72, 0.35], [0.72, 0.06]]]
            }
          }
        ]
      };
      handleDetectionResult(mockResult);
    } finally {
      setIsProcessing(false);
      setProcessingProgress('');
    }
  };

  return (
    <AuthWrapper>
      <div style={{padding:16, maxWidth: '1400px', margin: '0 auto'}}>
        <div style={{marginBottom: 24}}>
          <h1 style={{marginBottom: 8, color: '#2c3e50'}}>Room Detection AI</h1>
          <p style={{color: '#666', marginBottom: 8}}>
            Upload architectural blueprints and automatically detect room boundaries using AI
          </p>
          <div style={{fontSize: '12px', color: '#888'}}>
            <strong>Keyboard shortcuts:</strong> Ctrl+U (Upload), Ctrl+S (Saved), Ctrl+N (New File)
          </div>
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
                    background: '#d1ecf1',
                    border: '1px solid #bee5eb',
                    borderRadius: 4,
                    color: '#0c5460'
                  }}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                      <div style={{fontSize: '16px'}}>🔄</div>
                      <div>
                        <div style={{fontWeight: 'bold'}}>
                          Processing with {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}
                        </div>
                        <div style={{fontSize: '14px', marginTop: 2}}>
                          {processingProgress || 'Preparing...'}
                        </div>
                      </div>
                    </div>
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

