// Import THREE and OrbitControls from CDN via importmap
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

document.addEventListener('DOMContentLoaded', function () {
    const sceneContainer = document.getElementById('sceneContainer');
    const videoFileInput = document.getElementById('videoFile');
    const uploadButton = document.getElementById('uploadButton');
    const stopButton = document.getElementById('stopButton');
    const statusDiv = document.getElementById('status');
    const loader = document.getElementById('loader');

    let scene, camera, renderer, controls;
    let pathPoints = [];
    let pathLine;
    let asteroidSphere;
    let eventSource = null;
    let currentVideoPathForStream = null; // Track the video path for the current stream

    function initThreeJS() {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        camera = new THREE.PerspectiveCamera(75, sceneContainer.clientWidth / sceneContainer.clientHeight, 0.1, 2000);
        camera.position.set(50, 50, 150);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
        sceneContainer.innerHTML = ''; // Clear container before appending renderer
        sceneContainer.appendChild(renderer.domElement);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); // Increased intensity
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5); // Increased intensity
        directionalLight.position.set(5, 10, 7.5);
        scene.add(directionalLight);

        const axesHelper = new THREE.AxesHelper(100);
        scene.add(axesHelper);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 10;
        controls.maxDistance = 1000;
        controls.target.set(0,0,0);

        const material = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 }); // Added linewidth
        const geometry = new THREE.BufferGeometry();
        pathLine = new THREE.Line(geometry, material);
        scene.add(pathLine);

        const sphereGeometry = new THREE.SphereGeometry(2, 32, 32); // Smoother sphere
        const sphereMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000, emissive: 0x330000 }); // Added emissive
        asteroidSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        scene.add(asteroidSphere);
        asteroidSphere.visible = false;

        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    function updatePath(newPoint3D) {
        if (!newPoint3D || newPoint3D.length !== 3) {
            console.warn("Invalid 3D point received:", newPoint3D);
            return;
        }

        const vectorPoint = new THREE.Vector3(newPoint3D[0], newPoint3D[1], newPoint3D[2]);
        pathPoints.push(vectorPoint);

        const positions = new Float32Array(pathPoints.length * 3);
        for (let i = 0; i < pathPoints.length; i++) {
            positions[i * 3] = pathPoints[i].x;
            positions[i * 3 + 1] = pathPoints[i].y;
            positions[i * 3 + 2] = pathPoints[i].z;
        }
        pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        pathLine.geometry.attributes.position.needsUpdate = true; // Important
        pathLine.geometry.computeBoundingSphere();

        if (pathPoints.length > 0) {
            const lastPoint = pathPoints[pathPoints.length - 1];
            asteroidSphere.position.copy(lastPoint);
            asteroidSphere.visible = true;
        }
        if (pathPoints.length === 1) {
            controls.target.copy(pathPoints[0]);
            // Optional: Smart camera positioning for the first point
            // camera.position.copy(pathPoints[0]).add(new THREE.Vector3(30, 30, 50));
            controls.update();
        }
    }

    function resetVisualization() {
        pathPoints = [];
        if (pathLine && pathLine.geometry) {
            pathLine.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
            pathLine.geometry.attributes.position.needsUpdate = true;
            pathLine.geometry.computeBoundingSphere();
        }
        if (asteroidSphere) {
            asteroidSphere.visible = false;
        }
        if (controls) {
            controls.target.set(0,0,0);
            camera.position.set(50, 50, 150); // Reset camera position
            controls.update();
        }
    }


    uploadButton.addEventListener('click', async () => {
        const file = videoFileInput.files[0];
        if (!file) {
            statusDiv.textContent = 'Please select a video file first.';
            return;
        }

        // Force stop any existing stream before uploading a new video
        if (eventSource) {
            console.log("Closing existing event source before new upload.");
            eventSource.close();
            eventSource = null;
            // Also tell backend to stop its current processing explicitly
            await fetch('/stop_stream', { method: 'POST' });
        }

        resetVisualization();
        statusDiv.textContent = 'Uploading video...';
        loader.style.display = 'block';
        uploadButton.disabled = true;
        stopButton.disabled = true; // Keep disabled until stream starts

        const formData = new FormData();
        formData.append('videoFile', file);

        try {
            const response = await fetch('/upload_video', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (response.ok && result.video_path) {
                currentVideoPathForStream = result.video_path; // Store the path for the new stream
                statusDiv.textContent = 'Video uploaded. Starting processing and streaming...';
                startStreaming(currentVideoPathForStream);
                // stopButton will be enabled in startStreaming if successful
            } else {
                statusDiv.textContent = `Error: ${result.error || 'Upload failed'}`;
                loader.style.display = 'none';
                uploadButton.disabled = false;
            }
        } catch (error) {
            statusDiv.textContent = `Upload Error: ${error.message}`;
            loader.style.display = 'none';
            uploadButton.disabled = false;
        }
    });

    function startStreaming(videoPath) {
        if (!videoPath) {
            statusDiv.textContent = "No video path to stream.";
            return;
        }

        if (eventSource) { // Should have been closed by upload or stop, but as a safeguard
            eventSource.close();
        }
        console.log("Attempting to start stream for video: ", videoPath);
        eventSource = new EventSource(`/stream_3d_path?video_path=${encodeURIComponent(videoPath)}`);
        loader.style.display = 'block';
        statusDiv.textContent = 'Connecting to stream...';
        uploadButton.disabled = true; // Disable upload while stream is trying to connect/active
        stopButton.disabled = false; // Enable stop button now

        eventSource.onopen = function() {
            statusDiv.textContent = 'Streaming 3D path data...';
            loader.style.display = 'block'; // Ensure loader is visible
        };

        eventSource.onmessage = function (event) {
            try {
                const data = JSON.parse(event.data);

                if (data.error) {
                    console.error('Stream Error:', data.error);
                    statusDiv.textContent = `Stream Error: ${data.error}`;
                    stopAndCleanupStream();
                    return;
                }

                if (data.status) {
                    console.log('Stream Status:', data.status);
                    statusDiv.textContent = `Status: ${data.status}`;
                    if (data.status.toLowerCase().includes("complete") || data.status.toLowerCase().includes("closed")) {
                        stopAndCleanupStream(false); // Don't send another stop request if server initiated close
                    }
                    return;
                }

                if (data.point) {
                    updatePath(data.point);
                }
            } catch (e) {
                console.warn("Received non-JSON data or parse error:", event.data, e);
                // Potentially handle plain text status messages if any are sent outside JSON
            }
        };

        eventSource.onerror = function (error) {
            console.error('EventSource failed:', error);
            statusDiv.textContent = 'Error connecting to stream or stream interrupted.';
            stopAndCleanupStream(false); // Don't send stop request, connection already failed
        };
    }

    async function stopAndCleanupStream(sendStopRequest = true) {
        if (sendStopRequest) {
            try {
                await fetch('/stop_stream', { method: 'POST' });
                console.log("Stop request sent to backend.");
            } catch (e) {
                console.error("Error sending stop request:", e);
            }
        }
        if (eventSource) {
            eventSource.close();
            eventSource = null;
            console.log("EventSource closed.");
        }
        loader.style.display = 'none';
        uploadButton.disabled = false;
        stopButton.disabled = true;
        statusDiv.textContent = statusDiv.textContent + " Stream ended.";
        currentVideoPathForStream = null;
    }

    stopButton.addEventListener('click', () => {
        statusDiv.textContent = 'Stopping stream...';
        stopAndCleanupStream(true);
    });


    window.addEventListener('resize', () => {
        if (camera && renderer) {
            camera.aspect = sceneContainer.clientWidth / sceneContainer.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
        }
    });

    // Initial setup
    initThreeJS();
});