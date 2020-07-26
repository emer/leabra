Package `interinhib` provides inter-layer inhibition params, which can be added to Layer types.  Call at the start of the Layer InhibFmGeAct method like this:

```Go
// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.InterInhib.Inhib(&ly.Layer) // does inter-layer inhibition
	ly.PoolInhibFmGeAct(ltime)
}
```


