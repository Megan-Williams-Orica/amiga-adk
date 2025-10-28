// Socket.IO connection
const socket = io();

// D3.js plot variables
let svg, xScale, yScale, plotWidth, plotHeight;
let detectionSvg, detectionXScale, detectionYScale, detectionPlotWidth, detectionPlotHeight;

// ==================== Socket.IO Events ====================

socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
    addLog('Connected to navigation server', 'success');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
    addLog('Disconnected from server', 'error');
});

socket.on('status_update', (data) => {
    updateRobotStatus(data);
});

socket.on('success', (data) => {
    addLog(data.message, 'success');
});

socket.on('error', (data) => {
    addLog(data.message, 'error');
});

socket.on('warning', (data) => {
    addLog(data.message, 'warning');
});

socket.on('status', (data) => {
    addLog(data.message, 'info');
});

socket.on('nav_log', (data) => {
    // Navigation terminal output
    addLog('[NAV] ' + data.message, 'info');
});

// ==================== UI Update Functions ====================

function updateConnectionStatus(connected) {
    const badge = document.getElementById('connection-status');
    if (connected) {
        badge.textContent = 'Connected';
        badge.classList.remove('disconnected');
        badge.classList.add('connected');
    } else {
        badge.textContent = 'Disconnected';
        badge.classList.remove('connected');
        badge.classList.add('disconnected');
    }
}

function updateRobotStatus(status) {
    // Update pose
    if (status.pose) {
        document.getElementById('pos-x').textContent = status.pose.x.toFixed(2) + ' m';
        document.getElementById('pos-y').textContent = status.pose.y.toFixed(2) + ' m';
        document.getElementById('heading').textContent = status.pose.heading_deg.toFixed(1) + '°';
    }

    // Update waypoint
    document.getElementById('waypoint').textContent =
        `${status.current_waypoint} / ${status.total_waypoints}`;

    // Update track status
    const trackBadge = document.getElementById('track-status');
    trackBadge.textContent = status.track_status;

    // Update GPS quality
    const gpsBadge = document.getElementById('gps-quality');
    gpsBadge.textContent = status.gps_quality || 'UNKNOWN';

    // Update button states
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');

    if (status.navigation_running) {
        btnStart.disabled = true;
        btnStop.disabled = false;
    } else {
        btnStart.disabled = false;
        btnStop.disabled = true;
    }
}

function addLog(message, type = 'info') {
    const logContainer = document.getElementById('log-container');
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${timestamp}] ${message}`;
    logContainer.appendChild(entry);

    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;

    // Limit to 100 entries
    while (logContainer.children.length > 100) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

// ==================== Control Functions ====================

function startNavigation() {
    socket.emit('start_navigation');
    addLog('Sending start command...', 'info');
}

function stopNavigation() {
    socket.emit('stop_navigation');
    addLog('Sending stop command...', 'info');
}

function emergencyStop() {
    if (confirm('⚠️ EMERGENCY STOP - Are you sure?')) {
        socket.emit('emergency_stop');
        addLog('🛑 EMERGENCY STOP ACTIVATED', 'error');
    }
}

// ==================== Waypoint Plot (D3.js) ====================

function initializePlot() {
    const container = d3.select('#plot-container');
    const containerNode = container.node();
    const containerWidth = containerNode.clientWidth;
    const containerHeight = containerNode.clientHeight;

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    plotWidth = containerWidth - margin.left - margin.right;
    plotHeight = containerHeight - margin.top - margin.bottom;

    svg = container.append('svg')
        .attr('width', containerWidth)
        .attr('height', containerHeight)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Initial scales (will be updated with data)
    xScale = d3.scaleLinear().range([0, plotWidth]);
    yScale = d3.scaleLinear().range([plotHeight, 0]);

    // Add axes
    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${plotHeight})`)
        .style('color', '#9ca3af');

    svg.append('g')
        .attr('class', 'y-axis')
        .style('color', '#9ca3af');

    // Add axis labels
    svg.append('text')
        .attr('class', 'x-label')
        .attr('text-anchor', 'middle')
        .attr('x', plotWidth / 2)
        .attr('y', plotHeight + 35)
        .style('fill', '#9ca3af')
        .text('X (meters)');

    svg.append('text')
        .attr('class', 'y-label')
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('x', -plotHeight / 2)
        .attr('y', -45)
        .style('fill', '#9ca3af')
        .text('Y (meters)');
}

function updatePlot(data) {
    if (!svg) return;

    const waypoints = data.waypoints || [];
    const robot = data.robot;
    const currentIndex = data.current_index || 0;

    if (waypoints.length === 0) return;

    // Update scales
    const xExtent = d3.extent(waypoints, d => d.x);
    const yExtent = d3.extent(waypoints, d => d.y);

    // Add padding
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1 || 1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 1;

    xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);
    yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);

    // Update axes
    svg.select('.x-axis').call(d3.axisBottom(xScale).ticks(5));
    svg.select('.y-axis').call(d3.axisLeft(yScale).ticks(5));

    // Update waypoint markers
    const circles = svg.selectAll('.waypoint')
        .data(waypoints, d => d.index);

    circles.enter()
        .append('circle')
        .attr('class', 'waypoint')
        .merge(circles)
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', d => d.index === currentIndex ? 8 : 5)
        .attr('fill', d => {
            if (d.index === currentIndex) return '#3b82f6';
            if (d.index < currentIndex) return '#10b981';
            return '#ef4444';
        })
        .attr('stroke', d => d.index === currentIndex ? 'white' : 'none')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .append('title')
        .text(d => `Waypoint ${d.index}\nX: ${d.x.toFixed(2)}\nY: ${d.y.toFixed(2)}`);

    circles.exit().remove();

    // Update robot position
    if (robot && robot.x !== undefined && robot.y !== undefined) {
        let robotMarker = svg.select('.robot-marker');

        if (robotMarker.empty()) {
            robotMarker = svg.append('g').attr('class', 'robot-marker');

            // Robot body (circle)
            robotMarker.append('circle')
                .attr('r', 10)
                .attr('fill', '#f59e0b')
                .attr('stroke', 'white')
                .attr('stroke-width', 2);

            // Heading arrow
            robotMarker.append('path')
                .attr('class', 'heading-arrow')
                .attr('fill', 'white');
        }

        // Update robot position
        robotMarker.attr('transform',
            `translate(${xScale(robot.x)}, ${yScale(robot.y)})`);

        // Update heading arrow
        if (robot.heading !== undefined) {
            const arrowPath = `M 0,-12 L 4,0 L 0,8 L -4,0 Z`;
            robotMarker.select('.heading-arrow')
                .attr('d', arrowPath)
                .attr('transform', `rotate(${-robot.heading * 180 / Math.PI})`);
        }
    }
}

function fetchPlotData() {
    fetch('/plot_data')
        .then(response => response.json())
        .then(data => {
            console.log('Plot data received:', data);
            if (data.waypoints && data.waypoints.length > 0) {
                console.log(`Loaded ${data.waypoints.length} waypoints for plot`);
            } else {
                console.warn('No waypoints in plot data');
            }
            updatePlot(data);
        })
        .catch(error => {
            console.error('Error fetching plot data:', error);
            addLog(`Plot data error: ${error.message}`, 'error');
        });
}

// ==================== Detection Scatter Plot ====================

function initializeDetectionPlot() {
    const container = d3.select('#detection-plot-container');
    const containerNode = container.node();
    const containerWidth = containerNode.clientWidth;
    const containerHeight = containerNode.clientHeight;

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    detectionPlotWidth = containerWidth - margin.left - margin.right;
    detectionPlotHeight = containerHeight - margin.top - margin.bottom;

    detectionSvg = container.append('svg')
        .attr('width', containerWidth)
        .attr('height', containerHeight)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales: Y (left/right), X (forward)
    detectionXScale = d3.scaleLinear().domain([-5, 5]).range([0, detectionPlotWidth]);
    detectionYScale = d3.scaleLinear().domain([0, 10]).range([detectionPlotHeight, 0]);

    // Add axes
    detectionSvg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${detectionPlotHeight})`)
        .style('color', '#9ca3af')
        .call(d3.axisBottom(detectionXScale).ticks(5));

    detectionSvg.append('g')
        .attr('class', 'y-axis')
        .style('color', '#9ca3af')
        .call(d3.axisLeft(detectionYScale).ticks(5));

    // Add axis labels
    detectionSvg.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', detectionPlotWidth / 2)
        .attr('y', detectionPlotHeight + 35)
        .style('fill', '#9ca3af')
        .text('Y (meters) [left +, right -]');

    detectionSvg.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('x', -detectionPlotHeight / 2)
        .attr('y', -45)
        .style('fill', '#9ca3af')
        .text('X (meters) [forward +]');
}

function updateDetectionPlot(detections) {
    if (!detectionSvg || !detections) return;

    // Update detection markers
    const circles = detectionSvg.selectAll('.detection-marker')
        .data(detections, (d, i) => i);

    circles.enter()
        .append('circle')
        .attr('class', 'detection-marker')
        .merge(circles)
        .attr('cx', d => detectionXScale(d.y))  // Y = left/right
        .attr('cy', d => detectionYScale(d.x))  // X = forward
        .attr('r', 6)
        .attr('fill', '#a855f7')
        .attr('stroke', 'white')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .append('title')
        .text(d => `${d.label} (${(d.confidence * 100).toFixed(0)}%)\nDistance: ${d.distance.toFixed(2)}m`);

    circles.exit().remove();
}

function fetchDetectionData() {
    fetch('/detection_data')
        .then(response => response.json())
        .then(data => {
            updateDetectionPlot(data.detections);
        })
        .catch(error => {
            console.error('Error fetching detection data:', error);
        });
}

// ==================== Camera Feed ====================

function initializeCameraFeed() {
    const cameraFeed = document.getElementById('camera-feed');
    const cameraOverlay = document.getElementById('camera-overlay');

    cameraFeed.addEventListener('load', () => {
        // Hide "Waiting for camera..." message once feed loads
        if (cameraOverlay) {
            cameraOverlay.style.display = 'none';
        }
    });

    cameraFeed.addEventListener('error', () => {
        // Show error message if feed fails
        if (cameraOverlay) {
            cameraOverlay.style.display = 'block';
            cameraOverlay.innerHTML = '<div class="camera-info">Camera feed unavailable</div>';
        }
    });
}

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize camera feed
    initializeCameraFeed();

    // Initialize waypoint plot
    initializePlot();

    // Initialize detection scatter plot
    initializeDetectionPlot();

    // Fetch plot data every second
    setInterval(fetchPlotData, 1000);
    fetchPlotData();

    // Fetch detection data every second
    setInterval(fetchDetectionData, 1000);
    fetchDetectionData();

    // Initial log
    addLog('Dashboard loaded', 'info');
});
