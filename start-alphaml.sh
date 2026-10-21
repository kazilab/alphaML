#!/bin/sh
# Start alphaML 2.0 in Docker and open it in the web browser (macOS / Linux).
#   ./start-alphaml.sh          a running alphaML keeps its port, otherwise the first free port from 8765
#   ./start-alphaml.sh 8800     a specific port
# Stop it with:  docker compose down
cd "$(dirname "$0")" || exit 1

is_alphaml() { curl -fs --max-time 2 "http://127.0.0.1:$1/api/options" | grep -q '"AutoML"'; }
# curl exit code 7: nothing listens on the port
port_used() { curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/"; [ $? -ne 7 ]; }

docker version > /dev/null 2>&1 || { echo 'Docker is not running. Start Docker and try again.' >&2; exit 1; }

port="${1:-8765}"
running=$(docker port alphaml 8000 2> /dev/null | head -n 1 | sed 's/.*://')
if [ -z "$1" ] && [ -n "$running" ]; then
    port="$running"  # alphaML is already running: keep its port
elif ! is_alphaml "$port" && port_used "$port"; then
    if [ -n "$1" ]; then
        echo "Port $port is used by another program. Choose another one, e.g. ./start-alphaml.sh 8800" >&2
        exit 1
    fi
    while port_used "$port"; do port=$((port + 1)); done
    echo "Port 8765 is used by another program, using port $port."
fi

export ALPHAML_PORT="$port"
docker compose up -d --build || { echo 'docker compose failed, see the messages above.' >&2; exit 1; }

url="http://localhost:$port"
echo "Waiting for alphaML 2.0 at $url ..."
i=0
until is_alphaml "$port"; do
    i=$((i + 1))
    if [ "$i" -ge 120 ]; then
        echo "alphaML did not answer at $url. Check: docker compose ps   and   docker compose logs alphaml" >&2
        exit 1
    fi
    sleep 1
done
echo "alphaML 2.0 is running at $url  (stop it with: docker compose down)"
echo "Data sets and results: ${ALPHAML_DATA_DIR:-$(pwd)/tmp}"
if command -v open > /dev/null; then open "$url"; elif command -v xdg-open > /dev/null; then xdg-open "$url"; fi
