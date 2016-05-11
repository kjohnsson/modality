library(lattice)
library(RColorBrewer)

if (TRUE) {
  resdir <- '/Users/johnsson/Forskning/Experiments/modality/synthetic/'
} else {
  resdir <- '/home/johnsson/Forskning/Experiments/modality/synthetic/'
}
file <- 'summary_ref_shoulder_0.05.csv'  #'summary_0.05.csv'
data <- read.csv(paste0(resdir, file))
data$N = as.factor(data$N)
data$shoulder_variance = as.factor(data$shoulder_variance)

data <- data[order(data$shoulder_ratio),]
data <- data[order(data$shoulder_variance),]
data <- data[order(data$N),]

myColours <- brewer.pal(6,"Blues")

my.settings <- list(
  superpose.polygon=list(col=myColours[2:5], border="transparent"),
  strip.background=list(col=myColours[6]),
  strip.border=list(col="black")
)

# Varying shoulder variance (fixed weight 16/1)
barchart(frac_reject ~ N | shoulder_variance, groups=test,
         data=data[(data$ntest == 500) & (data$shoulder_ratio == " 16/1"), ], origin=0, 
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

library(beeswarm)
#ind = (data$ntest == 500) & (data$shoulder_ratio == " 16/1") &
#  (data$shoulder_variance == 0.0625)
#ind = ind & data$I == data[ind, 'I'][3]
beeswarm(frac_reject ~ N, data=data[, ],
         #pwpch=as.numeric(data[ind, 'I']))
         pwpch=as.numeric(data[, 'test']))

par(las=2, mar=c(7, 3, 1, 1))
beeswarm(frac_reject ~ test, data=data[, ],
         #pwpch=as.numeric(data[ind, 'I']))
         pwpch=as.numeric(data[, 'test']))

# Varying shoulder weight (fixed variance 1)
barchart(frac_reject ~ N | shoulder_ratio, groups=test,
         data=data[(data$ntest == 100) & (data$shoulder_variance == "1"), ], origin=0, 
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