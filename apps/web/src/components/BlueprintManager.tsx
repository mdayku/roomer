import React, { useState, useEffect } from 'react';

export interface SavedBlueprint {
  id: string;
  name: string;
  createdAt: Date;
  imageUrl: string;
  resultUrl?: string;
  status: 'processing' | 'completed' | 'error';
  modelUsed: string;
  roomCount?: number;
}

interface BlueprintManagerProps {
  blueprints: SavedBlueprint[];
  onSelect: (blueprint: SavedBlueprint) => void;
  onDelete: (id: string) => void;
  onDownload: (blueprint: SavedBlueprint) => void;
  selectedId?: string;
}

export const BlueprintManager: React.FC<BlueprintManagerProps> = ({
  blueprints,
  onSelect,
  onDelete,
  onDownload,
  selectedId
}) => {
  const [filter, setFilter] = useState<'all' | 'processing' | 'completed' | 'error'>('all');

  const filteredBlueprints = blueprints.filter(blueprint => {
    if (filter === 'all') return true;
    return blueprint.status === filter;
  });

  const getStatusColor = (status: SavedBlueprint['status']) => {
    switch (status) {
      case 'completed': return '#28a745';
      case 'processing': return '#ffc107';
      case 'error': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getStatusText = (status: SavedBlueprint['status']) => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'processing': return 'Processing';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  return (
    <div style={{
      border: '1px solid #dee2e6',
      borderRadius: 8,
      overflow: 'hidden',
      background: 'white'
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        background: '#f8f9fa',
        borderBottom: '1px solid #dee2e6',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold' }}>
          Saved Blueprints ({blueprints.length})
        </h3>

        {/* Filter */}
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as typeof filter)}
          style={{
            padding: '4px 8px',
            border: '1px solid #ccc',
            borderRadius: 4,
            fontSize: '14px'
          }}
        >
          <option value="all">All ({blueprints.length})</option>
          <option value="completed">
            Completed ({blueprints.filter(b => b.status === 'completed').length})
          </option>
          <option value="processing">
            Processing ({blueprints.filter(b => b.status === 'processing').length})
          </option>
          <option value="error">
            Error ({blueprints.filter(b => b.status === 'error').length})
          </option>
        </select>
      </div>

      {/* Blueprint List */}
      <div style={{
        maxHeight: '400px',
        overflowY: 'auto'
      }}>
        {filteredBlueprints.length === 0 ? (
          <div style={{
            padding: '40px 20px',
            textAlign: 'center',
            color: '#6c757d'
          }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>📄</div>
            <div>No blueprints found</div>
            <div style={{ fontSize: '14px', marginTop: '4px' }}>
              {filter === 'all' ? 'Upload your first blueprint!' : `No ${filter} blueprints`}
            </div>
          </div>
        ) : (
          filteredBlueprints.map((blueprint) => (
            <div
              key={blueprint.id}
              style={{
                padding: '12px 16px',
                borderBottom: '1px solid #f8f9fa',
                cursor: 'pointer',
                backgroundColor: selectedId === blueprint.id ? '#e3f2fd' : 'white',
                transition: 'background-color 0.2s'
              }}
              onClick={() => onSelect(blueprint)}
            >
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                {/* Thumbnail and info */}
                <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                  <img
                    src={blueprint.imageUrl}
                    alt={blueprint.name}
                    style={{
                      width: 60,
                      height: 45,
                      objectFit: 'cover',
                      borderRadius: 4,
                      marginRight: 12,
                      border: '1px solid #dee2e6'
                    }}
                  />

                  <div style={{ flex: 1 }}>
                    <div style={{
                      fontWeight: 'bold',
                      fontSize: '14px',
                      marginBottom: '2px',
                      color: '#333'
                    }}>
                      {blueprint.name}
                    </div>

                    <div style={{
                      fontSize: '12px',
                      color: '#666',
                      marginBottom: '4px'
                    }}>
                      {formatDate(blueprint.createdAt)}
                    </div>

                    <div style={{
                      fontSize: '12px',
                      color: '#666'
                    }}>
                      Model: {blueprint.modelUsed}
                      {blueprint.roomCount && ` • ${blueprint.roomCount} rooms detected`}
                    </div>
                  </div>
                </div>

                {/* Status and actions */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  {/* Status badge */}
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    color: 'white',
                    backgroundColor: getStatusColor(blueprint.status),
                    textTransform: 'uppercase'
                  }}>
                    {getStatusText(blueprint.status)}
                  </span>

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: '4px' }}>
                    {blueprint.status === 'completed' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDownload(blueprint);
                        }}
                        style={{
                          padding: '4px 8px',
                          border: '1px solid #007bff',
                          borderRadius: 3,
                          background: '#007bff',
                          color: 'white',
                          fontSize: '11px',
                          cursor: 'pointer'
                        }}
                        title="Download Results"
                      >
                        ↓
                      </button>
                    )}

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm('Delete this blueprint?')) {
                          onDelete(blueprint.id);
                        }
                      }}
                      style={{
                        padding: '4px 8px',
                        border: '1px solid #dc3545',
                        borderRadius: 3,
                        background: '#dc3545',
                        color: 'white',
                        fontSize: '11px',
                        cursor: 'pointer'
                      }}
                      title="Delete Blueprint"
                    >
                      ×
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
