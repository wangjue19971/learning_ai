import news_rec.code._matrixcf as _matrixcf

# load model
mcf = _matrixcf.MCF()
mcf.load_model()
print(mcf.imported)
test_ds = mcf.init_dataset("../data/test", is_train=False)
test_ds = iter(test_ds)

for i in range(3):
    ds = next(test_ds)
    label = ds.pop("ctr")
    print(ds)
    res = mcf.infer(ds)
    print(res)
