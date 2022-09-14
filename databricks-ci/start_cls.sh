#!/bin/bash

echo "Checking Cluster State (Cluster ID: ${cluster_id})..."
cluster_state=$(databricks clusters get --cluster-id "${cluster_id}" | jq -r ".state")
echo "Cluster State: $cluster_state"
if [ "$cluster_state" == "TERMINATED" ]; then
  echo "Starting Databricks Cluster..."
  databricks clusters start --cluster-id "${cluster_id}"
  sleep 30
  cluster_state=$(databricks clusters get --cluster-id "${cluster_id}" | jq -r ".state")
  echo "Cluster State: $cluster_state"
fi
while [ "$cluster_state" == "PENDING" ];
do
  sleep 30
  cluster_state=$(databricks clusters get --cluster-id "${cluster_id}" | jq -r ".state")
  echo "Cluster State: $cluster_state"
done
if [ "$cluster_state" == "RUNNING" ]; then
  exit 0
else
  exit 1
fi
