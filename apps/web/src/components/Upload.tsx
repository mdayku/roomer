import React, { useRef, useState } from 'react';
import { detectRooms } from '../lib/api';
import type { FeatureCollection } from '../types/geo';

export const Upload: React.FC<{
  onResult:(fc:FeatureCollection)=>void;
  onImageSelect?:(url:string)=>void;
  onFileUpload?:(file:File)=>Promise<void>;
  disabled?:boolean
}>=({ onResult, onImageSelect, onFileUpload, disabled = false })=>{
  const [busy,setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div style={{display:'flex',gap:12, alignItems:'center'}}>
      <input 
        ref={inputRef} 
        type="file" 
        accept="image/*" 
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file && onImageSelect) {
            const url = URL.createObjectURL(file);
            onImageSelect(url);
          }
        }}
      />
      <button
        disabled={busy || disabled}
        onClick={async()=>{
          try{
            setBusy(true);
            const file = inputRef.current?.files?.[0] ?? undefined;
            if (file && onFileUpload) {
              await onFileUpload(file);
            } else if (file) {
              const fc = await detectRooms(file);
              onResult(fc);
            }
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? 'Detecting...' : 'Detect Rooms'}
      </button>
    </div>
  );
};

