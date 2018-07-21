# Importing table qDiptab from R package diptest and write to file
# qDiptab.

tryCatch({
    library(diptest)
}, error = function(e) {
    library(utils)
    chooseCRANmirror(ind=1)
    install_packages('diptest')
    library(diptest)
})

data(qDiptab)
write.csv(qDiptab, 'modality/data/qDiptab.csv')
print("Tabluated p-values loaded from R package diptest.")
