pipeline {
    agent {
        label "purdue-cluster"
        }

    stages {
        stage('4.2-simulations-build'){
            steps{
                sh 'source /home/tgrogers-raid/a/common/gpgpu-sim-setup/4.2_env_setup.sh &&\
                make -j -C ./benchmarks/src all'
            }
        }
        }
    }
    post {
        always{
            emailext body: "See ${BUILD_URL}",
                recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                    [$class: 'RequesterRecipientProvider']],
                subject: "[AALP Jenkins] Build #${BUILD_NUMBER} - ${currentBuild.result}",
                to: 'tgrogers@purdue.edu'

        }
    }
}
