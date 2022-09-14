#!/bin/bash


JOB_ID=$(databricks jobs create --json "{
  \"name\": \"Testrun\",
  \"existing_cluster_id\": \"${cluster_id}\",
  \"timeout_seconds\": 3600,
  \"max_retries\": 1,
  \"notebook_task\": {
    \"notebook_path\": \"${notebook_folder}${notebook_name}\",
    \"base_parameters\": {}
  }
}" | jq ".job_id")

RUN_ID=$(databricks jobs run-now --job-id $JOB_ID | jq ".run_id")

job_status="PENDING"
while [ "$job_status" == "RUNNING" ] || [ "$job_status" == "PENDING" ]
do
  sleep 2
  job_status=$(databricks runs get --run-id $RUN_ID | jq -r ".state.life_cycle_state")
  echo Status $job_status
done

RESULT=$(databricks runs get-output --run-id $RUN_ID)

RESULT_STATE=$(echo $RESULT | jq -r ".metadata.state.result_state")
RESULT_MESSAGE=$(echo $RESULT | jq -r ".metadata.state.state_message")
if [ "$RESULT_STATE" == "FAILED" ]; then
  echo "Error message: $RESULT_MESSAGE"
fi

notebook_output=$(echo $RESULT | jq -r ".notebook_output.result")
echo $notebook_output
