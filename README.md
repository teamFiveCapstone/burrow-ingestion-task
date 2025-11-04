# An experiment using unstructured

To get started, just:

```
docker compose up
```

1. It should partition the `./files/wwf_tigers_e_1.pdf` and write the results to 
`./files/auto-partition-output.txt`. This is fast.

1. It should partition the `./files/wwf_tigers_e_1.pdf` again using the `partition_pdf` function using `hi-res` and write the results to 
`./files/pdf-partition-output.txt`. This takes a lot more time.

Then it should then create `./files/images` directory and put images from the PDF in there. I am actively trying to figure what to do with these