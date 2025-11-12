1.) Install aws cli(for mac)

```
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
```

```
sudo installer -pkg ./AWSCLIV2.pkg -target /
```

2.) Sign into aws account through aws cli

```
aws configure
```

3.) Connect docker login to aws ecr login

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ecr uri>
```

4.) Create docker image to upload to ecr

```
docker build -t quick-test --platform linux/amd64 .
```

5.) Connect docker image to ecr uri

```
docker tag quick-test:latest <ecr uri>
```

6.) Push image to ecr uri

```
docker push <ecr uri>
```

Random.) One step command after logged in for windows

```
docker buildx build --platform linux/arm64 -t <your_ecr_repo_url>:latest --push .
```
