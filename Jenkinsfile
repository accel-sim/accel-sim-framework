pipeline {
    agent {
        label "purdue-cluster"
        }

    options {
        disableConcurrentBuilds()
    }

    stages {
        stage('setup-env') {
            steps{
                sh 'rm -rf env-setup && git clone git@github.com:purdue-aalp/env-setup.git &&\
                    cd env-setup && git checkout cluster-ubuntu'
            }
        }
        stage('accel-sim-build'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/11.0_env_setup.sh
                source ./gpu-simulator/setup_environment.sh
                make -j -C gpu-simulator'''
            }
        }
        stage('rodinia_2.0-ft'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/11.0_env_setup.sh
                source ./gpu-simulator/setup_environment.sh
                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100 -T ~/../common/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft/9.1/ -N rodinia_2.0-ft-$$
                ./util/job_launching/monitor_func_test.py -I -v -s rodinia-stats-per-app.csv -N rodinia_2.0-ft-$$'''
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
