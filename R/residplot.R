# This is an example function named 'residplot'
# which generates Student residuals histogram

#' @title residplot
#' @description an example function  generating Student residuals histogram
#' @param fit the result of regression
#' @param nbreaks the number of group
#' @return a random sample of size \code{n}
#' @examples
#' states<-as.data.frame(state.x77[,c("Murder","Population","Illiteracy","Income","Frost")])
#' fit<-lm(Murder~ Population + Illiteracy + Income + Frost,data=states)
#' residplot(fit)
#' @export

residplot<- function(fit,nbreaks=10) {
  z<-rstudent(fit)
  x=seq(0,1,0.1)
  hist(z,breaks=nbreaks,freq = FALSE,xlab = "Student residual",main="Distributions of Errors")
  rug(jitter(z),col="brown")
  curve(dnorm(x,mean=mean(z),sd=sd(z)),add=TRUE,col="blue",lwd=2)
  lines(density(z)$x,density(z)$y,col="red",lwd=2,lty=2)
  legend("topright",legend=c("Normal Curve","Kernal Density Curve"),lty=1:2,col=c("blue","red"),cex=.7)
}
