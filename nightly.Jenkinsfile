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
                source ./env-setup/11.0_env_setup.sh
                rm -rf ./gpu-simulator/gpgpu-sim
                source ./gpu-simulator/setup_environment.sh
                make -j -C gpu-simulator'''
            }
        }
        stage('nightly-sass'){
            steps{
                sh '''#!/bin/bash -xe
                source ./env-setup/11.0_env_setup.sh
                source ./gpu-simulator/setup_environment.sh
                ./util/job_launching/run_simulations.py -B rodinia-3.1,GPU_Microbenchmark,sdk-4.2-scaled,parboil,polybench,cutlass_5_trace,Deepbench_nvidia_tencore,Deepbench_nvidia_normal -C QV100-SASS-5B_INSN -T ~/../common/accel-sim/traces/tesla-v100/latest/ -N nightly-$$ -M 70G
                ./util/job_launching/monitor_func_test.py -T 12 -S 1800 -I -v -s nightly-stats-per-app.csv -N nightly-$$'''
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
