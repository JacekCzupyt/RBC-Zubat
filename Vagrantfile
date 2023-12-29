DOCKER_BUILD_DIR = ENV['DOCKER_BUILD_DIR']
DOCKER_FILE_PATH = ENV['DOCKER_FILE_PATH']
DOCKER_IMAGE_URI = ENV['DOCKER_IMAGE_URI']

OUTPUT_DIR_HOST = ENV['OUTPUT_DIR_HOST']
OUTPUT_DIR_GUEST = ENV['OUTPUT_DIR_GUEST']
OUTPUT_FILENAME = ENV['OUTPUT_FILENAME']

CONVERT_TO_ENROOT = ENV['CONVERT_TO_ENROOT']


Vagrant.configure("2") do |config|
  config.vm.box = "hashicorp/bionic64"
  config.vm.network "forwarded_port", guest: 80, host: 2223
  config.vm.synced_folder "#{OUTPUT_DIR_HOST}", "#{OUTPUT_DIR_GUEST}"
  config.vm.provider "virtualbox" do |v|
    v.memory = 8096
    v.cpus = 8
  end

  config.vm.provision "docker" do |d|
    d.build_image "-t #{DOCKER_IMAGE_URI} -f #{DOCKER_FILE_PATH} #{DOCKER_BUILD_DIR}"
  end

  config.vm.post_up_message = "Docker image created"

  if CONVERT_TO_ENROOT == 'true' then
    config.vm.post_up_message = "Converting to enroot image"

    config.vm.provision "shell",
      path: "scripts/install_enroot.sh"

    config.vm.provision "shell",
      inline: "enroot import --output #{OUTPUT_DIR_GUEST}/#{OUTPUT_FILENAME} dockerd://#{DOCKER_IMAGE_URI}"

    config.vm.post_up_message = "Enroot image created"
  else
    config.vm.post_up_message = "Saving docker image"

    config.vm.provision "shell",
      inline: "docker save -o #{OUTPUT_DIR_GUEST}/#{OUTPUT_FILENAME} #{DOCKER_IMAGE_URI}"

    config.vm.post_up_message = "Docker image saved"
  end

end