cat << EOF > request.json
{
    "anthropic_version": "vertex-2023-10-16"
    ,"stream": false
    ,"max_tokens": 512
    ,"temperature": 0.5
    ,"top_p": 0.95
    ,"top_k": 1
    ,"messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Can you explain alignment faking?"
                }
            ]
        }
    ]
}
EOF

ENDPOINT="aiplatform.googleapis.com"
LOCATION_ID="us-east5"
PROJECT_ID="jieruan-gcp-bd84"
MODEL_ID="claude-3-5-haiku"
METHOD="rawPredict"

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @request.json \
"https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${LOCATION_ID}/publishers/anthropic/models/${MODEL_ID}:${METHOD}"