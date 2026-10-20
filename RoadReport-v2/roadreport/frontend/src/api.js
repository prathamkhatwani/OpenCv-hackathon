import axios from 'axios';
const api=axios.create({baseURL:'/api',timeout:300000});
api.interceptors.request.use(config=>{const key=sessionStorage.getItem('roadreport_key');if(key)config.headers['X-API-Key']=key;return config;});
export default api;
export const errorText=e=>typeof e.response?.data?.detail==='string'?e.response.data.detail:e.message||'Request failed';
