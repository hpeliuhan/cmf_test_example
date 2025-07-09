#!/bin/sh

docker build -t wifire-al:latest .
docker tag wifire-al:latest localhost:5000/wifire-al:latest
docker push localhost:5000/wifire-al:latest
kubectl delete deployment wifire-al
sleep 5
kubectl apply -f edge.yaml