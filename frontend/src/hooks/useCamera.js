import { useRef, useEffect, useCallback } from "react";

// MediaPipe doesn't need full resolution. Downscaling makes each frame smaller
// to encode, send, and decode, and makes pose estimation faster.
const DEFAULT_MAX_WIDTH = 480;

export function useCamera(onFrame, { fps = 30, maxWidth = DEFAULT_MAX_WIDTH, canSend } = {}) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      });
    return () => {
      clearInterval(intervalRef.current);
      if (videoRef.current?.srcObject)
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startCapture = useCallback(() => {
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      // Backpressure: if the backend is still working on the last frame,
      // skip this tick entirely instead of queueing a stale frame.
      if (canSend && !canSend()) return;

      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !video.videoWidth) return;

      const scale = Math.min(1, maxWidth / video.videoWidth);
      const w = Math.round(video.videoWidth * scale);
      const h = Math.round(video.videoHeight * scale);
      if (canvas.width !== w)  canvas.width  = w;
      if (canvas.height !== h) canvas.height = h;

      canvas.getContext("2d").drawImage(video, 0, 0, w, h);
      onFrame(canvas.toDataURL("image/jpeg", 0.7).split(",")[1]);
    }, 1000 / fps);
  }, [onFrame, fps, maxWidth, canSend]);

  const stopCapture = useCallback(() => clearInterval(intervalRef.current), []);
  return { videoRef, canvasRef, startCapture, stopCapture };
}