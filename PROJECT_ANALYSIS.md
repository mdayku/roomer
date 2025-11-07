# Project Analysis & Current State

## Overview

This document provides a comprehensive analysis of the Room Detection AI project, including the starter pack from ChatGPT, project requirements, existing assets, and the current implementation status.

## 1. Project Requirements (PRD)

### Core Goal
Automatically detect and return room boundaries from architectural blueprints (image or vector-derived raster). The output must support both rectangles and arbitrary polygons.

### Key Requirements
- **Input**: Blueprint images (PNG/JPG) or vector files
- **Output**: GeoJSON FeatureCollection with room polygons
- **Performance**: < 30 seconds per blueprint
- **Platform**: AWS (future), MVP runs locally
- **Stack**: React (frontend), Express/TypeScript (backend), AWS AI/ML services (future)

### Success Metrics
- **Efficiency**: 10-room plan processed in < 30s
- **Accuracy (MVP)**: ≥ 90% of rooms detected on simple plans; ≥ 75% IoU vs. human polygons on moderate plans
- **DX**: One-command local run; clear schemas & decision logs

## 2. Starter Pack Contents (from ChatGPT)

The `room_detection_ai_starter_pack_with_polygon_support.md` file contains:

### 2.1 Updated PRD
- Polygon-aware requirements
- Clear scope definition (MVP vs future)
- Success metrics and constraints

### 2.2 Data Schemas
- **GeoJSON-based output**: Standard format with normalized coordinates [0,1]
- **Feature structure**: Each room as a Feature with Polygon geometry
- **Properties**: id, name_hint, confidence, bbox_norm (optional)

### 2.3 API Design
- `POST /api/detect`: Accepts multipart form with image, returns GeoJSON
- `GET /api/health`: Health check endpoint

### 2.4 Architecture Diagram
Mermaid flowchart showing:
- User upload → React frontend → Express API → Detector Service → Response

### 2.5 Complete Codebase Scaffold
- Monorepo structure (pnpm workspace)
- Backend: Express + TypeScript with mock detector
- Frontend: React + Vite with upload and overlay components
- Type definitions for GeoJSON structures
- Geometry utilities (polygon closing, area calculation)

## 3. Existing Assets

### 3.1 CubiCasa5K Dataset
Located in `cubicasa5k/` directory:
- **~17,345 files** total
- **12,342 PNG files**: Blueprint images
- **5,000 SVG files**: Vector representations of floor plans
- **3 TXT files**: Dataset metadata/splits

### 3.2 COCO Format Annotations
Located in `cubicasa5k_coco/`:
- `train_coco_pt.json`: Training set annotations (COCO format)
- `val_coco_pt.json`: Validation set annotations
- `test_coco_pt.json`: Test set annotations

**Note**: COCO format is different from GeoJSON but can be converted. The COCO format includes:
- Image metadata (width, height, file_name)
- Annotations with segmentation masks/polygons
- Categories (room types)

### 3.3 Data Converters
Two converter scripts for different data sources:

**COCO Converter** (`apps/server/scripts/convert_coco_to_geojson.ts`):
- Converts COCO format annotations to GeoJSON
- Handles bounding boxes, polygons, and RLE masks
- Supports auto-detection of room categories
- Output: `cubicasa5k_geojson/` (49,522 rooms from 4,200 images)

**SVG Converter** (`apps/server/scripts/convert_cubicasa_svg_to_geojson.ts`):
- Converts CubiCasa SVG vector files to GeoJSON
- Supports polygon, polyline, and path elements
- Matches class names containing "space", "room", "area"
- Output: `cubicasa5k_svg_geojson/` (high-quality vector room polygons)

## 4. Current Implementation Status

### ✅ Completed
1. **Monorepo Structure**
   - Root package.json with workspace configuration
   - pnpm-workspace.yaml
   - Shared TypeScript base config

2. **Backend Server** (`apps/server/`)
   - Express API setup with CORS
   - `/api/detect` endpoint (mock implementation)
   - `/api/health` endpoint
   - Type definitions for GeoJSON
   - Geometry utilities
   - Mock detector service (returns seeded room polygons)

3. **Frontend Web App** (`apps/web/`)
   - React + Vite setup
   - Upload component for image selection
   - Overlay component for SVG polygon visualization
   - API client for backend communication
   - Type definitions matching backend

4. **Documentation**
   - README.md with setup instructions
   - Project structure documentation

5. **Utilities**
   - SVG to GeoJSON converter script (organized in scripts/)

### 🚧 Next Steps (Not Yet Implemented)

1. **Real Detection Logic**
   - Replace mock detector with actual AI/ML model
   - Options:
     - Classical CV: Canny edge detection + Hough transform → polygonization
     - ML model: Train on CubiCasa5K dataset
     - AWS services: Textract, SageMaker, Rekognition

2. **Image Processing**
   - Handle actual image uploads (currently mock ignores image)
   - Image preprocessing (resize, normalize)
   - Support for different image formats

3. **Data Pipeline**
   - Convert COCO annotations to GeoJSON ground truth
   - Dataset preparation for training
   - Validation metrics (IoU calculation)

4. **Frontend Enhancements**
   - Better image display (scaling, pan, zoom)
   - Interactive polygon editing
   - Confidence visualization
   - Room name hints display

5. **Testing**
   - Unit tests for geometry utilities
   - Integration tests for API
   - E2E tests for frontend

6. **AWS Integration**
   - Containerization (Docker)
   - S3 upload handling
   - Lambda/SageMaker inference skeleton
   - Environment configuration

## 5. Data Format Comparison

### Current Mock Output (GeoJSON)
```json
{
  "type": "FeatureCollection",
  "image": { "normalized": true },
  "features": [{
    "type": "Feature",
    "properties": {
      "id": "room_001",
      "name_hint": "Entry Hall",
      "confidence": 0.95,
      "bbox_norm": [0.05, 0.06, 0.18, 0.24]
    },
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[0.05,0.06], [0.18,0.06], [0.18,0.24], [0.05,0.24], [0.05,0.06]]]
    }
  }]
}
```

### CubiCasa COCO Format
- Uses segmentation masks or polygon coordinates
- Different coordinate system (pixel coordinates, not normalized)
- Includes category IDs and image references
- Needs conversion to match our GeoJSON schema

## 6. Technical Architecture

### Current Stack
- **Monorepo**: pnpm workspaces
- **Backend**: Express.js, TypeScript, Multer (file uploads)
- **Frontend**: React 18, Vite, TypeScript
- **Data Format**: GeoJSON (standard, tooling-friendly)

### Future Stack (AWS)
- **Compute**: Lambda or EC2/ECS
- **Storage**: S3 for blueprints
- **AI/ML**: SageMaker, Textract, or Rekognition
- **API**: API Gateway + Lambda or ECS

## 7. Development Workflow

### Current Setup
```bash
# Install dependencies
pnpm install

# Start development (both server and web)
pnpm dev

# Build for production
pnpm build

# Run tests
pnpm test
```

### Data Conversion
```bash
# Convert SVG files to GeoJSON
pnpm --filter @app/server tsx scripts/convert_cubicasa_svg_to_geojson.ts \
  --in ../../cubicasa5k \
  --out ../../cubicasa5k_geojson \
  --room-classes room Room Area Space
```

## 8. Key Decisions & Rationale

1. **GeoJSON Format**: Standard, well-supported, flexible (supports polygons, not just bboxes)
2. **Normalized Coordinates [0,1]**: Resolution-independent, simplifies scaling
3. **Monorepo**: Shared types, easier development, single install
4. **Mock First**: Allows frontend/backend development in parallel
5. **TypeScript**: Type safety, better DX, easier refactoring

## 9. Challenges & Considerations

1. **Dataset Format Mismatch**: COCO vs GeoJSON - need conversion pipeline
2. **Model Selection**: Classical CV vs ML - start simple, iterate
3. **Performance**: < 30s requirement - may need optimization/caching
4. **Accuracy**: Need validation against ground truth
5. **Polygon Complexity**: Support arbitrary shapes, not just rectangles

## 10. Recommendations

### Immediate (MVP)
1. ✅ Project structure is set up
2. Implement basic image upload handling
3. Add image display in frontend (currently missing)
4. Create simple classical CV detector (Canny + Hough)

### Short-term
1. Convert COCO annotations to GeoJSON ground truth
2. Implement IoU validation metrics
3. Train/test split from CubiCasa5K
4. Basic ML model (e.g., segmentation model)

### Long-term
1. AWS deployment
2. Production-grade error handling
3. Performance optimization
4. Advanced polygon refinement
5. Room name OCR integration

## 11. File Structure Summary

```
Roomer/
├── apps/
│   ├── server/                    # Backend API
│   │   ├── src/
│   │   │   ├── index.ts          # Express app entry
│   │   │   ├── routes/           # API routes
│   │   │   ├── services/         # Detection logic
│   │   │   ├── utils/            # Geometry utilities
│   │   │   └── types/            # TypeScript types
│   │   ├── scripts/              # Utility scripts
│   │   └── package.json
│   └── web/                       # Frontend React app
│       ├── src/
│       │   ├── components/       # React components
│       │   ├── lib/              # API client
│       │   └── types/            # TypeScript types
│       └── package.json
├── cubicasa5k/                    # Dataset (images + SVGs)
├── cubicasa5k_coco/              # COCO format annotations
├── package.json                   # Root workspace config
├── pnpm-workspace.yaml
├── tsconfig.base.json
├── README.md
└── PROJECT_ANALYSIS.md           # This file
```

## 12. Current Status Summary

### ✅ Completed Milestones
1. **Project Setup**: Monorepo structure with TypeScript backend/frontend
2. **Data Pipeline**: Two working converters for COCO and SVG data
3. **Training Data**: 49,522+ room polygons in GeoJSON format
4. **API Framework**: Express server with mock detection endpoint
5. **Frontend**: React app with image upload and polygon visualization

### 🚧 Next Steps (Ready for Implementation)
1. **Real Detection Algorithm**: Replace mock with actual room detection
2. **ML Model Training**: Use converted GeoJSON data for training
3. **Performance Optimization**: Implement < 30s processing requirement
4. **AWS Integration**: Containerize and deploy to cloud

### 📊 Data Assets Available
- **Ground Truth**: 49,522 room polygons from COCO annotations
- **Vector Data**: High-quality SVG-derived room polygons
- **Mock System**: Working end-to-end pipeline for testing

### 🛠 Technical Stack Ready
- **Backend**: Express + TypeScript + GeoJSON processing
- **Frontend**: React + Vite + SVG visualization
- **Data**: Multiple sources converted to unified GeoJSON format
- **Tools**: Conversion scripts, geometry utilities, API framework

The foundation is solid. The project is ready for implementing the actual room detection algorithm using the extensive training data now available in both GeoJSON and COCO formats. The COCO format with polygon segmentation is particularly well-suited for training Mask R-CNN and other instance segmentation models.

---

**Last Updated**: Cloud YOLO training pipeline complete - AWS SageMaker ready for 2-4h training
**Status**: ✅ Ready for cloud training - YOLO on SageMaker recommended (costs covered by Gauntlet AI)

