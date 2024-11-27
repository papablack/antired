import {v4 as uuidGen} from 'uuid';

export const uniqid = () : string => {
    return uuidGen();
}