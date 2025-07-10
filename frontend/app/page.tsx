"use client";

import { useRef, useEffect, useState } from "react";

type PredictionResult = {
  recieved: string;
  confidence: number;
  prediction: string;
};

export default function Home() {
  // BIG IDEA: get users webcam stream -> get curr frame & send to backend -> get and process prediction
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  // connect to ws
  const socketRef = useRef<WebSocket>(null);

  function drawFrameToCanvas(
    canvas: HTMLCanvasElement,
    video: HTMLVideoElement,
  ) {
    canvas.height = video.videoHeight;
    canvas.width = video.videoWidth;

    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  }

  // send frame data from canvas (raw binary) to backend
  function sendFrameData(canvas: HTMLCanvasElement): void {
    // console.log("sending frame...");
    const socket = socketRef.current;
    // console.log(canvas.toDataURL("image/jpeg"));
    canvas.toBlob((frameData: Blob | null) => {
      // ensure websocket connection, then send message
      if (!frameData) return;
      if (socket && socket.readyState == socket.OPEN) {
        socket.send(frameData);
      }
    }, "image/jpeg");
  }

  function processFrame(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
    drawFrameToCanvas(canvas, video);
    sendFrameData(canvas);
  }

  // handle socket response
  function handlePredictionResult({ data }: MessageEvent) {
    const prediction: PredictionResult = JSON.parse(data);
    setPrediction(prediction);
  }

  // socket listeners
  useEffect(() => {
    socketRef.current = new WebSocket("ws://localhost:8000/predict");
    const socket = socketRef.current;

    if (socket) {
      socket.onopen = () => console.log("socket connected!");
      socket.onmessage = handlePredictionResult;
    }

    return () => socket?.close();
  }, []);

  // start video stream from user's webcam
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
  }, []);

  // get camera frame & send it to backend @ N fps
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    let interval: NodeJS.Timeout;
    const N = 10;

    if (canvas && video) {
      interval = setInterval(() => processFrame(canvas, video), 1000 / N);
    }

    return () => clearInterval(interval);
  }, []);

  return (
    <main>
      <video ref={videoRef} className="-scale-x-100" autoPlay />
      <span>
        <p className="text-3xl font-bold">{prediction?.prediction}</p>
        <p>{prediction?.confidence}</p>
      </span>
      <canvas ref={canvasRef} className="hidden" />
    </main>
  );
}
