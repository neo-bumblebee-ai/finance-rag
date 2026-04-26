#!/usr/bin/env bash
# deploy.sh — Build, push, and deploy Finance RAG to AWS ECS Fargate
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure)
#   - Docker installed and running
#   - Secrets already created in Secrets Manager (see README)
#
# Usage:
#   ./deploy/aws/deploy.sh [environment]     # environment = production | staging
#
# Environment variables (set in CI or export before running):
#   AWS_ACCOUNT_ID      — your 12-digit AWS account ID
#   AWS_REGION          — target region (default: us-east-1)
#   CERTIFICATE_ARN     — ACM certificate ARN for HTTPS
#   OPENAI_SECRET_ARN   — Secrets Manager ARN for OPENAI_API_KEY
#   COHERE_SECRET_ARN   — Secrets Manager ARN for COHERE_API_KEY
#   JWT_SECRET_ARN      — Secrets Manager ARN for JWT_SECRET_KEY
#   LANGSMITH_SECRET_ARN— Secrets Manager ARN for LANGCHAIN_API_KEY
#   LANGFUSE_SECRET_ARN — Secrets Manager ARN for Langfuse keys (JSON)

set -euo pipefail

ENV=${1:-production}
REGION=${AWS_REGION:-us-east-1}
ACCOUNT=${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID not set}
PROJECT="finance-rag"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT}"
IMAGE_TAG=$(git rev-parse --short HEAD)
STACK_NAME="${PROJECT}-${ENV}"

echo "========================================"
echo "  Finance RAG Deployment"
echo "  Environment : ${ENV}"
echo "  Region      : ${REGION}"
echo "  Image tag   : ${IMAGE_TAG}"
echo "========================================"

# ── 1. Build Docker image ─────────────────────────────────────────────────
echo "[1/5] Building Docker image..."
docker build -t "${PROJECT}:${IMAGE_TAG}" .

# ── 2. Push to ECR ────────────────────────────────────────────────────────
echo "[2/5] Authenticating with ECR and pushing image..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"

docker tag "${PROJECT}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker tag "${PROJECT}:${IMAGE_TAG}" "${ECR_URI}:latest"
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"
echo "  Pushed: ${ECR_URI}:${IMAGE_TAG}"

# ── 3. Deploy / update CloudFormation stack ───────────────────────────────
echo "[3/5] Deploying CloudFormation stack: ${STACK_NAME}..."
aws cloudformation deploy \
  --region "${REGION}" \
  --template-file deploy/aws/cloudformation.yml \
  --stack-name "${STACK_NAME}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName="${PROJECT}" \
    Environment="${ENV}" \
    ContainerImage="${ECR_URI}:${IMAGE_TAG}" \
    CertificateArn="${CERTIFICATE_ARN:?}" \
    OpenAISecretArn="${OPENAI_SECRET_ARN:?}" \
    CohereSecretArn="${COHERE_SECRET_ARN:?}" \
    JwtSecretArn="${JWT_SECRET_ARN:?}" \
    LangSmithSecretArn="${LANGSMITH_SECRET_ARN:?}" \
    LangfuseSecretArn="${LANGFUSE_SECRET_ARN:?}" \
  --no-fail-on-empty-changeset

# ── 4. Force ECS to use new image ─────────────────────────────────────────
echo "[4/5] Forcing new ECS deployment..."
CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" \
  --output text)

aws ecs update-service \
  --cluster "${CLUSTER}" \
  --service "${PROJECT}-service" \
  --force-new-deployment \
  --region "${REGION}" \
  --output table

# ── 5. Wait for stability ─────────────────────────────────────────────────
echo "[5/5] Waiting for service to stabilize (up to 5 min)..."
aws ecs wait services-stable \
  --cluster "${CLUSTER}" \
  --services "${PROJECT}-service" \
  --region "${REGION}"

ALB_URL=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ALBURL'].OutputValue" \
  --output text)

echo ""
echo "========================================"
echo "  Deployment complete!"
echo "  API URL: ${ALB_URL}"
echo "  Health:  ${ALB_URL}/health"
echo "  Docs:    ${ALB_URL}/docs"
echo "========================================"
