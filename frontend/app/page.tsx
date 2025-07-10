"use client";
import { useRef, useEffect } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // connect to ws
  const socket = new WebSocket("ws://localhost:8000/predict");
  socket.onopen = () => console.log("socket connected!");

  function handleVideoStream(frameData: Blob | null): void {
    // send frame data (raw binary) to backend
    // ensure websocket connection, then send message
    // handle result
  }

  function getFrame(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
    canvas.height = video.videoHeight;
    canvas.width = video.videoWidth;

    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    ctx.restore();

    // get raw binary for frame
    canvas.toBlob(handleVideoStream, "image/jpeg");
  }

  useEffect(() => {
    async function setVideoStream(video: HTMLVideoElement) {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 226 * 6, height: 406 * 2 },
        audio: false,
      });

      video.srcObject = stream;
    }

    const video = videoRef.current;
    if (video) {
      setVideoStream(video);
    }

    const canvas = canvasRef.current;
    if (canvas && video) {
      getFrame(canvas, video);
    }
  }, []);

  return (
    <main>
      <video ref={videoRef} className="-scale-x-100" autoPlay />
      <canvas ref={canvasRef} />
    </main>
  );
}
