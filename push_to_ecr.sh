#!/bin/bash

# --- CONFIGURE THESE ---
AWS_ACCOUNT_ID=517589119834
AWS_REGION=us-east-1
REPO_NAME=gradio-app
IMAGE_TAG=latest
# ------------------------

set -e

echo "🚧 Building Docker image..."
docker build -t $REPO_NAME .

echo "🔖 Tagging image for ECR..."
docker tag $REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG

echo "🔐 Authenticating Docker with ECR..."
aws ecr get-login-password --region $AWS_REGION | \
docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "🚀 Pushing image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG

echo "✅ Done! Image pushed to ECR:"
echo "   $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
