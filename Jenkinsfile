pipeline {
    agent any

    environment {
        DOCKER_HUB_CREDENTIALS = '78ef0425-3b9c-4b5c-a282-11a38457cf72'  // Jenkins credentials ID
        IMAGE_NAME = 'aymanaomarihub/flask'
        IMAGE_TAG = 'v1.0.3'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Docker Login') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
            }
        }

        stage('Push Docker Image') {
            steps {
                sh 'docker push $IMAGE_NAME:$IMAGE_TAG'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([file(credentialsId: 'k3d-mycluster-kubeconfig', variable: 'KUBECONFIG')]) {
                    sh 'kubectl set image deployment/flask-app flask-app=$IMAGE_NAME:$IMAGE_TAG'
                    sh 'kubectl rollout status deployment/flask-app'
                }
            }
        }
    }

    post {
        success {
            echo "✅ Successfully pushed image $IMAGE_NAME:$IMAGE_TAG to Docker Hub."
        }
        failure {
            echo "❌ Build failed. Check the logs."
        }
    }
}