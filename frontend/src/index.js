import React from 'react';
import ReactDOM from 'react-dom';
import PersistentDrawer from './Navbar';
import registerServiceWorker from './registerServiceWorker';

ReactDOM.render(<PersistentDrawer />, document.getElementById('root'));
registerServiceWorker();
