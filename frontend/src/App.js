import React, { Component } from 'react';
import { withStyles } from '@material-ui/core/styles';

import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';

import Grow from '@material-ui/core/Grow';

import TextField from '@material-ui/core/TextField';

import IconButton from '@material-ui/core/IconButton';
import Icon from '@material-ui/core/Icon';
import Tooltip from '@material-ui/core/Tooltip';

import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';

import LinearProgress from '@material-ui/core/LinearProgress';

import 'typeface-roboto'

const styles = theme => ({
  button: {
    margin: theme.spacing.unit,
  },
  paper: {
    padding: theme.spacing.unit * 2,
    textAlign: 'center',
    color: theme.palette.text.secondary,
  },
  text: {
    'text-transform': 'capitalize'
  }
});

class App extends Component {
  constructor(props){
    super(props);
    this.state = {
      text: '',
      prediction: '',
      showResults: false,
      predictionEmpty: true
    };
  }
  handleSubmit(e){
    e.preventDefault();
    if (!this.state.text.length) {
      return;
    }
    this.setState({showResults: true})
    fetch('call', {
      method: 'post',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data: this.state.text
      })
    })
    .then(result=>result.json())
    .then(item=>this.setState({prediction: item.prediction, showResults: false, predictionEmpty:false}))    
  }
  handleChange(e) {
    this.setState({ text: e.target.value });
  }
  SentimentIcon(props) {
    const state = props.state;
    if(state.predictionEmpty == false){
      if(state.prediction == 'positive'){
        return (<Tooltip id="tooltip-bottom" title="positive" placement="top"><Icon>sentiment_very_satisfied</Icon></Tooltip>);
      } else {
        return (<Tooltip id="tooltip-bottom" title="negative" placement="top"><Icon>sentiment_very_dissatisfied</Icon></Tooltip>);
      }
   }
   return null;
  }
  
  render(){
    const {classes} = this.props;
    return (
      <React.Fragment>
          <Grid container spacing={24} justify="center">
            <Grid item xs={12} sm={6}>
              <Grow in={true} timeout={1000} >
                <Paper className={classes.paper}>
                  <form className={classes.container} onSubmit={(e) => this.handleSubmit(e)} noValidate autoComplete="off">
                    <TextField multiline fullWidth
                      id="text"
                      label="Say Something"
                      margin="normal"
                      onChange={(e) => this.handleChange(e)}
                      value={this.state.text}
                      />
                    <IconButton type="submit" className={classes.button} aria-label="send" >
                      <Icon>send</Icon>
                    </IconButton>
                  </form>
                  {this.state.showResults ? <LinearProgress /> : null}
                </Paper>
              </Grow>
            </Grid>
          </Grid>
          <Grid container spacing={24} justify='center'>
            <Grid item xm={12} sm={6}>
              <Grow in={true} timeout={2000} >
                <Paper className={classes.paper}>
                  <List>
                      <ListItem>
                        <ListItemText className={classes.text}
                          primary="Prediction"
                          secondary={this.state.prediction}
                        />          
                        
                        <this.SentimentIcon state={this.state}/>
                      </ListItem>
                    </List>
                </Paper>
              </Grow>
            </Grid>
          </Grid>
      </React.Fragment>
    )
  };
  
}


export default withStyles(styles)(App);
