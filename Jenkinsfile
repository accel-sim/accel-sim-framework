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
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                rm -rf ./gpu-simulator/gpgpu-sim
                source ./gpu-simulator/setup_environment.sh
                make -j -C gpu-simulator
                make clean -C gpu-simulator
                make -j -C gpu-simulator'''
            }
        }
        stage('short-test'){
            steps{
                parallel "sass": {
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                source ./gpu-simulator/setup_environment.sh
                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft,GPU_Microbench -C QV100-SASS -T ~/../common/accel-sim/traces/tesla-v100/latest/ -N sass-short-$$
                ./util/job_launching/monitor_func_test.py -I -v -s stats-per-app-sass.csv -N sass-short-$$'''
               }, "ptx": {
                sh '''#!/bin/bash -xe
                source ./env-setup/11.2.1_env_setup.sh
                source ./gpu-simulator/setup_environment.sh

                rm -rf ./gpu-app-collection
                git clone git@github.com:accel-sim/gpu-app-collection.git
                source ./gpu-app-collection/src/setup_environment
                make rodinia_2.0-ft GPU_Microbench -j -C ./gpu-app-collection/src
                ./gpu-app-collection/get_regression_data.sh

                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft,GPU_Microbench -C QV100-PTX -N short-ptx-$$
                ./util/job_launching/monitor_func_test.py -I -v -s stats-per-app-ptx.csv -N short-ptx-$$'''
               }
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
