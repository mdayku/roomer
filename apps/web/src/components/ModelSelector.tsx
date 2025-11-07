import React from 'react';

export interface Model {
  id: string;
  name: string;
  description: string;
  status: 'available' | 'training' | 'error';
  accuracy?: number;
  inferenceTime?: number;
}

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (modelId: string) => void;
  models: Model[];
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  models
}) => {
  const getStatusColor = (status: Model['status']) => {
    switch (status) {
      case 'available': return '#28a745';
      case 'training': return '#ffc107';
      case 'error': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getStatusText = (status: Model['status']) => {
    switch (status) {
      case 'available': return 'Available';
      case 'training': return 'Training';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  };

  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{
        display: 'block',
        marginBottom: 8,
        fontWeight: 'bold',
        color: '#333'
      }}>
        Select Detection Model:
      </label>

      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        style={{
          width: '100%',
          padding: '8px 12px',
          border: '1px solid #ccc',
          borderRadius: 4,
          fontSize: '14px',
          backgroundColor: 'white'
        }}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name} - {getStatusText(model.status)}
            {model.accuracy && ` (${model.accuracy.toFixed(1)}% accuracy)`}
            {model.inferenceTime && ` - ${model.inferenceTime}ms`}
          </option>
        ))}
      </select>

      {/* Model details */}
      {selectedModel && (
        <div style={{
          marginTop: 8,
          padding: 12,
          background: '#f8f9fa',
          borderRadius: 4,
          border: '1px solid #dee2e6'
        }}>
          {(() => {
            const model = models.find(m => m.id === selectedModel);
            if (!model) return null;

            return (
              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: 8
                }}>
                  <span
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      backgroundColor: getStatusColor(model.status),
                      marginRight: 8
                    }}
                  />
                  <strong>{model.name}</strong>
                  <span style={{
                    marginLeft: 8,
                    padding: '2px 6px',
                    background: getStatusColor(model.status),
                    color: 'white',
                    borderRadius: 3,
                    fontSize: '12px'
                  }}>
                    {getStatusText(model.status)}
                  </span>
                </div>

                <p style={{ margin: 0, color: '#666', fontSize: '14px' }}>
                  {model.description}
                </p>

                {model.accuracy && (
                  <p style={{
                    margin: '4px 0 0 0',
                    color: '#666',
                    fontSize: '13px'
                  }}>
                    Accuracy: {model.accuracy.toFixed(1)}% |
                    Inference: {model.inferenceTime || 'N/A'}ms
                  </p>
                )}
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};
