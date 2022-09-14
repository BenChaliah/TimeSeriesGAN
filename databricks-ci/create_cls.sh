#!/bin/bash

echo "Create a new cluster"
databricks clusters create --json '{"cluster_name": "myCluster", "spark_version": "10.5.x-cpu-ml-scala2.12", "node_type_id": "Standard_DS3_v2", "autoscale" : { "min_workers": 2, "max_workers": 50 } }'