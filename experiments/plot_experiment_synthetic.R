library(lattice)
library(RColorBrewer)

data <- read.csv('/home/johnsson/Forskning/Experiments/modality/synthetic/summary_0.05.csv')
data$N = as.factor(data$N)

myColours <- brewer.pal(6,"Blues")

my.settings <- list(
  superpose.polygon=list(col=myColours[2:5], border="transparent"),
  strip.background=list(col=myColours[6]),
  strip.border=list(col="black")
)

barchart(frac_reject ~ N | shoulder_ratio, groups=test,
         data=data[data$ntest == 100, ], origin=0, 
         main="Rejection ratio", 
         xlab="N", ylab="Proportion of rejected tests",
         scales=list(alternating=1),                  
         auto.key=list(space="top", columns=4, 
                       points=FALSE, rectangles=TRUE,
                       title="Test type", cex.title=1),
         par.settings = my.settings,
         par.strip.text=list(col="white", font=2),
         panel=function(x,y,...){
           panel.grid(h=-1, v=0); 
           panel.barchart(x,y,...)
         }
)