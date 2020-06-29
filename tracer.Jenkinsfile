pipeline {
    agent {
        label "dgx-gpu01"
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
        stage('build-tracer'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/10.1_env_setup.sh
                ./util/tracer_nvbit/install_nvbit.sh
                make clean -C ./util/tracer_nvbit/
                make -C ./util/tracer_nvbit/'''
            }
        }
        stage('rodinia_2.0-ft-build'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/10.1_env_setup.sh
                rm -rf ./gpu-app-collection/
                git clone git@github.com:accel-sim/gpu-app-collection.git
                source ./gpu-app-collection/src/setup_environment
                make -C ./gpu-app-collection/src rodinia_2.0-ft'''
            }
        }
        stage('generate-rodinia_2.0-ft-traces'){
            steps{
                '''#!/bin/bash
                source ./env-setup/10.1_env_setup.sh
                rm -rf ./run_hw/
                ./run_hw_trace.py -B rodinia_2.0-ft -D 7
                '''
            }
        }
        stage('accel-sim-build'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/10.1_env_setup.sh
                rm -rf ./gpu-simulator/gpgpu-sim
                source ./gpu-simulator/setup_environment.sh
                make -j -C gpu-simulator'''
            }
        }
        stage('test-new-traces'){
            steps{
                sh '''#!/bin/bash
                source ./env-setup/10.1_env_setup.sh
                source ./gpu-simulator/setup_environment.sh
                ./util/job_launching/run_simulations.py -B rodinia_2.0-ft -C QV100 -T ./run_hw/traces/device-7/10.1/ -N rodinia_2.0-ft-$$
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
