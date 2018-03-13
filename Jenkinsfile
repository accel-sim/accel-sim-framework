pipeline {
    agent {
        label "purdue-cluster"
        }

    stages {
        stage('4.2-simulations-build'){
            steps{
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                source ./benchmarks/src/setup_environment &&\
                make -C ./benchmarks/src clean_rodinia-3.1 &&\
                make -C ./benchmarks/src all'
            }
        }
        stage('9.1-simulations-build'){
            steps{
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                source ./benchmarks/src/setup_environment && \
                make -C ./benchmarks/src clean_rodinia-3.1 && \
                make -C ./benchmarks/src all'
            }
        }
        stage('9.1-simulations-build'){
            steps{
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/9.1_env_setup.sh &&\
                source ./benchmarks/src/setup_environment && \
                make -C ./benchmarks/src clean_rodinia-3.1 && \
                make -C ./benchmarks/src all'
            }
        }
    }
    post {
        success {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - Success!",
                to: 'tgrogers@purdue.edu'
        }
        failure {
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - ${currentBuild.result}",
                to: 'tgrogers@purdue.edu'
        }
   }
}
